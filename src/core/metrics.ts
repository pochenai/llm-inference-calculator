// End-to-end evaluation: VRAM check -> TTFT / TPOT -> E2E latency + throughput.
//
// evaluate() is the public entry point and NEVER throws: it returns a
// Result<EvaluationResult> so a UI can branch on r.ok and render r.error.

import type { Calibration } from './calibration';
import { IDEAL } from './calibration';
import { CalcError, err, ok } from './errors';
import type { Result } from './errors';
import type { ParallelLayout, SystemSpec, Workload } from './types';
import { DEFAULT_PREFILL_RATIO } from './types';
import { deriveConstants } from './model';
import { resolveInterconnect, peakFlopsOf } from './hardware';
import { validateLayout } from './layout';
import type { VramBreakdown } from './memory';
import { PD_DECODE, PD_PREFILL, vramBreakdown } from './memory';
import type { DecodeDetail, PrefillDetail, SdDecodeDetail } from './latency';
import { buildCommModel, decodeStepTime, prefillTime, sdDecodeStepTime } from './latency';

export interface EvaluationResult {
  feasible: boolean;
  memory: VramBreakdown; // headline breakdown (decode pool in disagg mode)
  memoryPrefillPool?: VramBreakdown; // only in PD-disaggregation mode
  prefill: PrefillDetail;
  decode: DecodeDetail; // SD-adjusted when speculative enabled
  kvTransferExposedMs: number; // PD-disaggregation KV transfer, overlap-adjusted
  ttftMs: number;
  tpotMs: number; // SD-adjusted (lower than baseline when SD enabled)
  e2eMs: number; // SD-adjusted
  throughputTps: number; // output tokens per second, system level (SD-adjusted)
  // Steady-state workload: effective batch sizes per pool
  prefillBatchSize: number; // r * batchSize (PD) or batchSize (non-PD)
  decodeBatchSize: number; // (1-r) * batchSize (PD) or batchSize (non-PD)
  // Optimal PD pool GPU allocation for pipeline rate matching.
  // optimalPrefillFraction = N_p / (N_p + N_d) such that prefill output rate
  // equals decode drain rate. optimalPrefillGpus = round(optimalPrefillFraction * totalGpus).
  // Only meaningful when PD disaggregation is enabled.
  optimalPrefillFraction?: number;
  optimalPrefillGpus?: number;
  // Resource utilization metrics
  prefillComputeUtilization: number; // fraction of time spent on compute (tComputeMs / ttftMs)
  decodeBandwidthUtilization: number; // fraction of time spent on bandwidth (tBandwidthMs / tpotMs)
  prefillActualFlops: number; // actual FLOPS achieved (in TFLOPS)
  prefillPeakFlops: number; // hardware peak FLOPS (in TFLOPS)
  decodeActualBandwidth: number; // actual bandwidth achieved (in GB/s)
  decodePeakBandwidth: number; // hardware peak bandwidth (in GB/s)
  // Speculative decoding details (present only when SD enabled)
  speculative?: {
    draftModelId: string;
    draftModelName: string;
    gamma: number;
    acceptanceRate: number;
    draftStepMs: number;
    verifyStepMs: number;
    cycleTimeMs: number;
    expectedTokensPerCycle: number;
    baselineTpotMs: number;
    baselineE2eMs: number;
    baselineThroughputTps: number;
    speedup: number;
    // Note: draftWeightsBytes and draftKvBytes are in memory.draftWeightsBytes and memory.draftKvBytes
  };
}

// Public entry point: never throws; all failures surface as { ok: false }.
export function evaluate(spec: SystemSpec, cal: Calibration = IDEAL): Result<EvaluationResult> {
  try {
    return ok(evaluateInner(spec, cal));
  } catch (e) {
    if (e instanceof CalcError) return err(e);
    return err(new CalcError('internal', e instanceof Error ? e.message : String(e)));
  }
}

function evaluateInner(spec: SystemSpec, cal: Calibration): EvaluationResult {
  const derived = deriveConstants(spec.model, spec.weightQuant, spec.kvQuant);
  const inter = resolveInterconnect(spec.gpu, spec.interNodeBwGbps, spec.intraNodeBwGbps);
  const comm = buildCommModel(inter, cal);
  const memOpts = { flashAttention: spec.flashAttention, headroom: spec.headroom };

  // Steady-state workload ratio: split total batch into prefill and decode pools.
  // r = prefill fraction, (1-r) = decode fraction.
  // When prefillRatio is undefined, the steady-state contention model is
  // disabled: no batch splitting, no queuing delay between phases.
  const steadyState = spec.workload.prefillRatio !== undefined;
  const r = steadyState
    ? Math.max(0.1, Math.min(0.9, spec.workload.prefillRatio!))
    : DEFAULT_PREFILL_RATIO;
  const oneMinusR = 1 - r;

  const layoutErrors: string[] = [];
  let memory: VramBreakdown;
  let memoryPrefillPool: VramBreakdown | undefined;
  let prefillLayout: ParallelLayout;
  let decodeLayout: ParallelLayout;
  let prefillGpus: number;
  let decodeGpus: number;
  let kvTransferExposedMs = 0;
  let prefillBatchSize: number;
  let decodeBatchSize: number;
  let optimalPrefillFraction: number | undefined;
  let optimalPrefillGpus: number | undefined;

  // Workloads with pool-specific batch sizes for PD disaggregation.
  let prefillWorkload: Workload;
  let decodeWorkload: Workload;

  // Speculative decoding: derive constants for draft model
  let draftDerived: ReturnType<typeof deriveConstants> | undefined;
  if (spec.speculative) {
    draftDerived = deriveConstants(spec.speculative.draftModel, spec.weightQuant, spec.kvQuant);
  }

  if (spec.disagg) {
    const d = spec.disagg;
    prefillLayout = d.prefillLayout;
    decodeLayout = d.decodeLayout;
    prefillGpus = d.prefillGpus;
    decodeGpus = d.decodeGpus;
    layoutErrors.push(...validateLayout(prefillLayout, spec.model, prefillGpus, spec.gpusPerNode));
    layoutErrors.push(...validateLayout(decodeLayout, spec.model, decodeGpus, spec.gpusPerNode));

    // Split batch by workload ratio for steady-state PD modeling.
    // When steady-state is disabled, both pools use the full batch.
    prefillBatchSize = steadyState ? Math.max(1, Math.round(r * spec.workload.batchSize)) : spec.workload.batchSize;
    decodeBatchSize = steadyState ? Math.max(1, Math.round(oneMinusR * spec.workload.batchSize)) : spec.workload.batchSize;
    prefillWorkload = { ...spec.workload, batchSize: prefillBatchSize };
    decodeWorkload = { ...spec.workload, batchSize: decodeBatchSize };

    memoryPrefillPool = vramBreakdown(
      spec.model,
      derived,
      spec.gpu,
      prefillLayout,
      prefillWorkload,
      { ...memOpts, pdMode: PD_PREFILL },
    );
    // SD-aware VRAM for decode pool (steady-state KV: inputLen + outputLen/2)
    memory = vramBreakdown(
      spec.model,
      derived,
      spec.gpu,
      decodeLayout,
      decodeWorkload,
      { ...memOpts, pdMode: PD_DECODE },
      spec.speculative?.draftModel,
      draftDerived,
      spec.speculative?.draftTp,
    );

    // KV cache shipped from prefill pool to decode pool after prefill,
    // at sequence length N_in. Only the prefill batch's KV is transferred.
    const kvBytes = derived.kv.totalBytes(spec.workload.inputLen) * prefillBatchSize;
    const kvTransferMs = (kvBytes / comm.interBwBps) * 1e3;
    kvTransferExposedMs = kvTransferMs * (1 - d.kvTransferOverlap);
  } else {
    prefillLayout = spec.layout;
    decodeLayout = spec.layout;
    prefillGpus = spec.layout.tp * spec.layout.ep * spec.layout.pp * spec.layout.dp;
    decodeGpus = prefillGpus;
    layoutErrors.push(...validateLayout(spec.layout, spec.model, prefillGpus, spec.gpusPerNode));
    // Non-PD: same batch for both phases (time-multiplexed on same GPUs).
    prefillBatchSize = spec.workload.batchSize;
    decodeBatchSize = spec.workload.batchSize;
    prefillWorkload = spec.workload;
    decodeWorkload = spec.workload;
    // VRAM: when the solver passes a pdMode hint, use pool-specific seqLen
    // (prefill = inputLen only, decode = inputLen + outputLen/2) instead of
    // the default non-PD average. Without the hint, standard non-PD sizing.
    const solverPd = spec.solverPdMode;
    memory = vramBreakdown(
      spec.model,
      derived,
      spec.gpu,
      spec.layout,
      spec.workload,
      solverPd ? { ...memOpts, pdMode: solverPd } : memOpts,
      spec.speculative?.draftModel,
      draftDerived,
      spec.speculative?.draftTp,
    );
  }

  if (layoutErrors.length > 0) {
    throw new CalcError('invalid-layout', `Invalid layout: ${layoutErrors.join('; ')}`);
  }

  // Prefill phase: use pool-specific workload for PD, original for non-PD.
  const prefillPhaseBase = {
    model: spec.model,
    derived,
    gpu: spec.gpu,
    workload: prefillWorkload,
    weightQuant: spec.weightQuant,
    cal,
    comm,
    gpusPerNode: spec.gpusPerNode,
  };

  const prefill = prefillTime({ ...prefillPhaseBase, layout: prefillLayout }, prefillGpus);

  // Decode phase: use pool-specific workload for PD.
  const decodePhaseBase = {
    model: spec.model,
    derived,
    gpu: spec.gpu,
    workload: decodeWorkload,
    weightQuant: spec.weightQuant,
    cal,
    comm,
    gpusPerNode: spec.gpusPerNode,
  };

  let decode: DecodeDetail;
  let tpotMs: number;
  let speculativeResult: EvaluationResult['speculative'] | undefined;

  if (spec.speculative && draftDerived) {
    // Speculative decoding path
    const draftInp = {
      model: spec.speculative.draftModel,
      derived: draftDerived,
      gpu: spec.gpu,
      layout: { tp: spec.speculative.draftTp, pp: 1, ep: 1, dp: 1 } as ParallelLayout,
      workload: decodeWorkload,
      weightQuant: spec.weightQuant,
      cal,
      comm,
      gpusPerNode: spec.gpusPerNode,
    };

    const sdDetail: SdDecodeDetail = sdDecodeStepTime(
      { ...decodePhaseBase, layout: decodeLayout },
      draftInp,
      spec.speculative.gamma,
      spec.speculative.acceptanceRate,
    );

    // decode field uses verifyStep (main model's decode detail)
    decode = sdDetail.verifyStep;
    tpotMs = sdDetail.tpotMs;

    // Baseline metrics (without SD) for comparison
    const baselineTpotMs = sdDetail.baselineTpotMs;
    const baselineE2eMs = prefill.ttftMs + kvTransferExposedMs + spec.workload.outputLen * baselineTpotMs;
    const baselineThroughputTps =
      (decodeBatchSize * spec.workload.outputLen) / (baselineE2eMs / 1e3);

    speculativeResult = {
      draftModelId: spec.speculative.draftModel.id,
      draftModelName: spec.speculative.draftModel.name,
      gamma: spec.speculative.gamma,
      acceptanceRate: spec.speculative.acceptanceRate,
      draftStepMs: sdDetail.draftStep.tpotMs,
      verifyStepMs: sdDetail.verifyStep.tpotMs,
      cycleTimeMs: sdDetail.cycleTimeMs,
      expectedTokensPerCycle: sdDetail.expectedTokensPerCycle,
      baselineTpotMs,
      baselineE2eMs,
      baselineThroughputTps,
      speedup: sdDetail.speedup,
    };
  } else {
    // Standard decode path
    decode = decodeStepTime({ ...decodePhaseBase, layout: decodeLayout });
    tpotMs = decode.tpotMs;
  }

  const feasible = memory.feasible && (memoryPrefillPool ? memoryPrefillPool.feasible : true);

  // Non-PD resource contention: prefill and decode share the same GPUs.
  // In steady state, the GPU alternates between prefill and decode work.
  // The GPU time fraction for each phase determines the probability that
  // a new operation must wait for the other phase to finish.
  let ttftMs: number;
  let effectiveTpotMs: number;
  if (spec.disagg) {
    // PD: pools are physically separate, no contention.
    ttftMs = prefill.ttftMs + kvTransferExposedMs;
    effectiveTpotMs = tpotMs;
  } else if (!steadyState) {
    // Steady-state contention model disabled: pure phase times, no queuing.
    ttftMs = prefill.ttftMs;
    effectiveTpotMs = tpotMs;
  } else {
    // Non-PD: compute GPU occupancy fractions from total work per cycle.
    // prefill_throughput = B * N_in / TTFT (tokens/s when GPU is doing prefill)
    // decode_throughput = B / TPOT (tokens/s when GPU is doing decode)
    const B = spec.workload.batchSize;
    const nIn = spec.workload.inputLen;
    const nOut = spec.workload.outputLen;
    const prefillThroughput = prefill.ttftMs > 0 ? (B * nIn) / (prefill.ttftMs / 1e3) : 0;
    const decodeThroughput = tpotMs > 0 ? B / (tpotMs / 1e3) : 0;
    // Total GPU time per steady-state cycle (arbitrary observation window):
    const tPrefillTotal = prefillThroughput > 0 ? (r * B * nIn) / prefillThroughput : 0;
    const tDecodeTotal = decodeThroughput > 0 ? (oneMinusR * B * nOut) / decodeThroughput : 0;
    const tCycle = tPrefillTotal + tDecodeTotal;
    const rhoPrefill = tCycle > 0 ? tPrefillTotal / tCycle : 0;
    const rhoDecode = tCycle > 0 ? tDecodeTotal / tCycle : 0;
    // TTFT: prefill time + expected wait from decode contention.
    // The prefill randomly lands at any position within the decode phase
    // (which lasts tpotMs * N_out). Average wait = 1/2 the phase duration.
    ttftMs = prefill.ttftMs + rhoDecode * tpotMs * nOut / 2;
    // TPOT: decode step time + expected delay from prefill contention.
    // Each decode step randomly lands at any position within the prefill
    // phase (which lasts prefill.ttftMs). Average wait = 1/2 the phase.
    effectiveTpotMs = tpotMs + rhoPrefill * prefill.ttftMs / 2;
  }

  const e2eMs = ttftMs + spec.workload.outputLen * effectiveTpotMs;

  // Throughput: for PD, use decode batch size (only decode pool produces output
  // tokens). For non-PD, use total batch size (all requests share GPUs).
  const throughputBatch = spec.disagg ? decodeBatchSize : spec.workload.batchSize;
  const throughputTps = (throughputBatch * spec.workload.outputLen) / (e2eMs / 1e3);

  // Optimal PD GPU allocation for pipeline rate matching.
  // Balance constraint: r·B / T_prefill = (1-r)·B / (N_out · T_step)
  // => DP_p/DP_d = T_prefill / (N_out · T_step) · (1-r)/r
  // With identical per-replica layouts, GPU ratio = DP ratio:
  //   x = TTFT / (N_out · TPOT) · (1-r)/r
  //   ρ = N_p/(N_p+N_d) = x/(1+x) = TTFT / (TTFT + ((1-r)/r) · N_out · TPOT)
  if (spec.disagg && tpotMs > 0) {
    const decodeTotalMs = spec.workload.outputLen * tpotMs;
    const ratio = oneMinusR / r; // (1-r)/r factor from workload split
    optimalPrefillFraction = ttftMs / (ttftMs + ratio * decodeTotalMs);
    const totalPdGpus = prefillGpus + decodeGpus;
    optimalPrefillGpus = Math.max(1, Math.min(totalPdGpus - 1, Math.round(optimalPrefillFraction * totalPdGpus)));
  }

  // Calculate utilization metrics based on hardware peak values (not calibrated)
  // Prefill: actual FLOPS/s / hardware peak FLOPS/s
  const numGpus = spec.disagg
    ? spec.disagg.prefillGpus // Use prefill pool GPUs for prefill calculation
    : spec.layout.tp * spec.layout.ep * spec.layout.pp * spec.layout.dp;
  const peakFlopsPerGpu = peakFlopsOf(spec.gpu, spec.weightQuant);
  const totalPeakFlops = numGpus * peakFlopsPerGpu;
  const actualFlopsPerSec = ttftMs > 0 ? prefill.flops / (ttftMs / 1e3) : 0;
  const prefillComputeUtilization = totalPeakFlops > 0 ? actualFlopsPerSec / totalPeakFlops : 0;
  // Convert to TFLOPS for display
  const prefillActualFlops = actualFlopsPerSec / 1e12;
  const prefillPeakFlops = totalPeakFlops / 1e12;

  // Decode: actual bytes/s / hardware peak bytes/s
  const actualBytesPerSec = tpotMs > 0 ? decode.bytesPerStep / (tpotMs / 1e3) : 0;
  const peakBytesPerSec = spec.gpu.bwGbps * 1e9;
  const decodeBandwidthUtilization = peakBytesPerSec > 0 ? actualBytesPerSec / peakBytesPerSec : 0;
  // Convert to GB/s for display
  const decodeActualBandwidth = actualBytesPerSec / 1e9;
  const decodePeakBandwidth = peakBytesPerSec / 1e9;

  const result: EvaluationResult = {
    feasible,
    memory,
    prefill,
    decode,
    kvTransferExposedMs,
    ttftMs,
    tpotMs: effectiveTpotMs, // includes non-PD prefill contention overhead
    e2eMs,
    throughputTps,
    prefillBatchSize,
    decodeBatchSize,
    prefillComputeUtilization,
    decodeBandwidthUtilization,
    prefillActualFlops,
    prefillPeakFlops,
    decodeActualBandwidth,
    decodePeakBandwidth,
  };
  if (memoryPrefillPool) result.memoryPrefillPool = memoryPrefillPool;
  if (optimalPrefillFraction !== undefined) result.optimalPrefillFraction = optimalPrefillFraction;
  if (optimalPrefillGpus !== undefined) result.optimalPrefillGpus = optimalPrefillGpus;
  if (speculativeResult) result.speculative = speculativeResult;
  return result;
}
