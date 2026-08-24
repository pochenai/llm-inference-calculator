// End-to-end evaluation: VRAM check -> TTFT / TPOT -> E2E latency + throughput.
//
// evaluate() is the public entry point and NEVER throws: it returns a
// Result<EvaluationResult> so a UI can branch on r.ok and render r.error.

import type { Calibration } from './calibration';
import { IDEAL } from './calibration';
import { CalcError, err, ok } from './errors';
import type { Result } from './errors';
import type { ParallelLayout, SystemSpec } from './types';
import { deriveConstants } from './model';
import { resolveInterconnect, peakFlopsOf } from './hardware';
import { validateLayout } from './layout';
import type { VramBreakdown } from './memory';
import { vramBreakdown } from './memory';
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

  const layoutErrors: string[] = [];
  let memory: VramBreakdown;
  let memoryPrefillPool: VramBreakdown | undefined;
  let prefillLayout: ParallelLayout;
  let decodeLayout: ParallelLayout;
  let prefillGpus: number;
  let decodeGpus: number;
  let kvTransferExposedMs = 0;

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

    memoryPrefillPool = vramBreakdown(
      spec.model,
      derived,
      spec.gpu,
      prefillLayout,
      spec.workload,
      memOpts,
    );
    // SD-aware VRAM for decode pool
    memory = vramBreakdown(
      spec.model,
      derived,
      spec.gpu,
      decodeLayout,
      spec.workload,
      memOpts,
      spec.speculative?.draftModel,
      draftDerived,
      spec.speculative?.draftTp,
    );

    // KV cache shipped from prefill pool to decode pool after prefill,
    // at sequence length N_in.
    // Only main model's KV is transferred; draft model does its own prefill
    // in the decode pool (small enough to be negligible).
    const kvBytes = derived.kv.totalBytes(spec.workload.inputLen) * spec.workload.batchSize;
    const kvTransferMs = (kvBytes / comm.interBwBps) * 1e3;
    kvTransferExposedMs = kvTransferMs * (1 - d.kvTransferOverlap);
  } else {
    prefillLayout = spec.layout;
    decodeLayout = spec.layout;
    prefillGpus = spec.layout.tp * spec.layout.ep * spec.layout.pp * spec.layout.dp;
    decodeGpus = prefillGpus;
    layoutErrors.push(...validateLayout(spec.layout, spec.model, prefillGpus, spec.gpusPerNode));
    // SD-aware VRAM
    memory = vramBreakdown(
      spec.model,
      derived,
      spec.gpu,
      spec.layout,
      spec.workload,
      memOpts,
      spec.speculative?.draftModel,
      draftDerived,
      spec.speculative?.draftTp,
    );
  }

  if (layoutErrors.length > 0) {
    throw new CalcError('invalid-layout', `Invalid layout: ${layoutErrors.join('; ')}`);
  }

  const phaseBase = {
    model: spec.model,
    derived,
    gpu: spec.gpu,
    workload: spec.workload,
    weightQuant: spec.weightQuant,
    cal,
    comm,
    gpusPerNode: spec.gpusPerNode,
  };

  const prefill = prefillTime({ ...phaseBase, layout: prefillLayout }, prefillGpus);

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
      workload: spec.workload,
      weightQuant: spec.weightQuant,
      cal,
      comm,
      gpusPerNode: spec.gpusPerNode,
    };

    const sdDetail: SdDecodeDetail = sdDecodeStepTime(
      { ...phaseBase, layout: decodeLayout },
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
      (spec.workload.batchSize * spec.workload.outputLen) / (baselineE2eMs / 1e3);

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
    decode = decodeStepTime({ ...phaseBase, layout: decodeLayout });
    tpotMs = decode.tpotMs;
  }

  const feasible = memory.feasible && (memoryPrefillPool ? memoryPrefillPool.feasible : true);

  const ttftMs = prefill.ttftMs + kvTransferExposedMs;
  const e2eMs = ttftMs + spec.workload.outputLen * tpotMs;
  const throughputTps = (spec.workload.batchSize * spec.workload.outputLen) / (e2eMs / 1e3);

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
    tpotMs,
    e2eMs,
    throughputTps,
    prefillComputeUtilization,
    decodeBandwidthUtilization,
    prefillActualFlops,
    prefillPeakFlops,
    decodeActualBandwidth,
    decodePeakBandwidth,
  };
  if (memoryPrefillPool) result.memoryPrefillPool = memoryPrefillPool;
  if (speculativeResult) result.speculative = speculativeResult;
  return result;
}
