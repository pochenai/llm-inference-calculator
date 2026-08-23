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
import { resolveInterconnect } from './hardware';
import { validateLayout } from './layout';
import type { VramBreakdown } from './memory';
import { vramBreakdown } from './memory';
import type { DecodeDetail, PrefillDetail } from './latency';
import { buildCommModel, decodeStepTime, prefillTime } from './latency';

export interface EvaluationResult {
  feasible: boolean;
  memory: VramBreakdown; // headline breakdown (decode pool in disagg mode)
  memoryPrefillPool?: VramBreakdown; // only in PD-disaggregation mode
  prefill: PrefillDetail;
  decode: DecodeDetail;
  kvTransferExposedMs: number; // PD-disaggregation KV transfer, overlap-adjusted
  ttftMs: number;
  tpotMs: number;
  e2eMs: number;
  throughputTps: number; // output tokens per second, system level
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

  if (spec.disagg) {
    const d = spec.disagg;
    prefillLayout = d.prefillLayout;
    decodeLayout = d.decodeLayout;
    prefillGpus = d.prefillGpus;
    decodeGpus = d.decodeGpus;
    layoutErrors.push(...validateLayout(prefillLayout, spec.model, prefillGpus, spec.gpusPerNode));
    layoutErrors.push(...validateLayout(decodeLayout, spec.model, decodeGpus, spec.gpusPerNode));

    memoryPrefillPool = vramBreakdown(spec.model, derived, spec.gpu, prefillLayout, spec.workload, memOpts);
    memory = vramBreakdown(spec.model, derived, spec.gpu, decodeLayout, spec.workload, memOpts);

    // KV cache shipped from prefill pool to decode pool after prefill,
    // at sequence length N_in.
    const kvBytes = derived.kv.totalBytes(spec.workload.inputLen) * spec.workload.batchSize;
    const kvTransferMs = (kvBytes / comm.interBwBps) * 1e3;
    kvTransferExposedMs = kvTransferMs * (1 - d.kvTransferOverlap);
  } else {
    prefillLayout = spec.layout;
    decodeLayout = spec.layout;
    prefillGpus = spec.layout.tp * spec.layout.ep * spec.layout.pp * spec.layout.dp;
    decodeGpus = prefillGpus;
    layoutErrors.push(...validateLayout(spec.layout, spec.model, prefillGpus, spec.gpusPerNode));
    memory = vramBreakdown(spec.model, derived, spec.gpu, spec.layout, spec.workload, memOpts);
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
  const decode = decodeStepTime({ ...phaseBase, layout: decodeLayout });

  const feasible = memory.feasible && (memoryPrefillPool ? memoryPrefillPool.feasible : true);

  const ttftMs = prefill.ttftMs + kvTransferExposedMs;
  const tpotMs = decode.tpotMs;
  const e2eMs = ttftMs + spec.workload.outputLen * tpotMs;
  const throughputTps = (spec.workload.batchSize * spec.workload.outputLen) / (e2eMs / 1e3);

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
  };
  if (memoryPrefillPool) result.memoryPrefillPool = memoryPrefillPool;
  return result;
}
