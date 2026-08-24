// Latency model: TTFT (prefill, compute axis) and TPOT (decode, bandwidth axis).
// All time quantities are expressed in milliseconds (ms).

import type { Calibration } from './calibration';
import { exposedComm } from './calibration';
import type { GpuSpec, Interconnect, ModelSpec, ParallelLayout, Workload } from './types';
import { B_ACT } from './types';
import type { DerivedConstants } from './model';
import { attentionQuadFlops } from './model';
import { peakFlopsOf } from './hardware';
import { shardFactors } from './layout';

export interface CommModel {
  intraBwBps: number; // effective bandwidth, bytes/s (comm efficiency applied)
  interBwBps: number; // bytes/s
  alphaIntraMs: number; // measured small-message collective latency (total, not per-hop)
  alphaInterMs: number;
}

export function buildCommModel(inter: Interconnect, cal: Calibration): CommModel {
  return {
    intraBwBps: inter.intraNodeBwGbps * 1e9 * cal.commEffIntra,
    interBwBps: inter.interNodeBwGbps * 1e9 * cal.commEffInter,
    alphaIntraMs: cal.alphaIntraMs,
    alphaInterMs: cal.alphaInterMs,
  };
}

// One ring all-reduce call, in ms:
//   bandwidth term 2(t-1)/t * msg / BW (reduce-scatter + all-gather)
//   latency term   alpha_collective (measured total; negligible for large msg)
export function ringAllreduceMs(msgBytes: number, tp: number, comm: CommModel): number {
  if (tp <= 1) return 0;
  const bandwidthTermMs = (((2 * (tp - 1)) / tp) * msgBytes / comm.intraBwBps) * 1e3;
  return bandwidthTermMs + comm.alphaIntraMs;
}

// One all-to-all call (EP dispatch or combine), in ms.
export function allToAllMs(sendBytes: number, ep: number, comm: CommModel): number {
  if (ep <= 1) return 0;
  return (sendBytes / comm.intraBwBps) * 1e3 + comm.alphaIntraMs;
}

export interface PrefillDetail {
  flops: number;
  tComputeMs: number; // ideal compute time
  tCommMs: number; // total comm time (TP all-reduce + EP all-to-all + PP P2P)
  ttftMs: number;
}

export interface PhaseInput {
  model: ModelSpec;
  derived: DerivedConstants;
  gpu: GpuSpec;
  layout: ParallelLayout;
  workload: Workload;
  weightQuant: Parameters<typeof peakFlopsOf>[1];
  cal: Calibration;
  comm: CommModel;
  gpusPerNode: number; // used to pick intra- vs inter-node bandwidth for PP P2P
}

export function prefillTime(inp: PhaseInput, numGpus: number): PrefillDetail {
  const { model, derived, gpu, layout, workload, cal, comm } = inp;
  const B = workload.batchSize;
  const nIn = workload.inputLen;

  const flopsMatmul = B * nIn * derived.flopsPerToken;
  const flopsAttn = attentionQuadFlops(model, derived.kv, nIn) * B;
  const flops = flopsMatmul + flopsAttn;

  const clusterFlops = numGpus * peakFlopsOf(gpu, inp.weightQuant) * cal.mfuPrefill;
  const tComputeMs = (flops / clusterFlops) * 1e3;
  // NOTE: PP pipeline bubble is intentionally NOT modeled (see README assumption);
  // real schedulers minimize it, and a (B+pp-1)/B estimate was too pessimistic.

  // TP: 2 all-reduces per layer (after attention output, after MLP output).
  const msgBytes = B * nIn * model.hiddenSize * B_ACT;
  const tTpMs = model.layers * 2 * ringAllreduceMs(msgBytes, layout.tp, comm);

  // EP: dispatch + combine per MoE layer (v1: all layers of a MoE model are MoE).
  let tEpMs = 0;
  if (model.type === 'moe' && model.moe && layout.ep > 1) {
    const buffer = B * nIn * model.moe.expertsPerToken * model.hiddenSize * B_ACT;
    const sendBytes = (buffer * (layout.ep - 1)) / layout.ep;
    tEpMs = model.layers * 2 * allToAllMs(sendBytes, layout.ep, comm);
  }

  // PP: point-to-point activation transfer across (pp-1) stage boundaries.
  // The batch flows as B microbatches; each boundary link carries all B of
  // them, each message = one sequence's hidden state [N_in, h]. The (pp-1)
  // boundary links run in parallel, so the first-order makespan cost is the
  // bytes on a single boundary link. Pipeline fill/drain (bubble) is not
  // modeled; per-hop P2P latency is a lower-order term omitted here.
  let tPpMs = 0;
  if (layout.pp > 1) {
    const ppMsgBytes = B * nIn * model.hiddenSize * B_ACT;
    // Heuristic: a multi-node deployment is assumed to run PP across nodes.
    const ppBwBps = numGpus > inp.gpusPerNode ? comm.interBwBps : comm.intraBwBps;
    tPpMs = (ppMsgBytes / ppBwBps) * 1e3;
  }

  const tCommMs = tTpMs + tEpMs + tPpMs;
  const exposedTpMs = exposedComm(tTpMs, cal.tpCommOverlap);
  const exposedEpMs = exposedComm(tEpMs, cal.epCommOverlap);
  const exposedPpMs = exposedComm(tPpMs, cal.ppCommOverlap);
  const ttftMs = tComputeMs + exposedTpMs + exposedEpMs + exposedPpMs;
  return { flops, tComputeMs, tCommMs, ttftMs };
}

// Fraction of the expert set that must be read from HBM in a single decode step.
// Each of the B tokens routes to k experts; P(a given expert is never picked)
// = (1 - k/E)^B, so coverage = 1 - (1 - k/E)^B.
//
// This is a hard per-step lower bound on expert weight reads, NOT reducible by
// pipelining: the step must produce outputs for all B tokens before advancing,
// so every expert hit by >= 1 token must be read at least once within the step.
// Overlap/pipelining can hide the read latency but cannot reduce the byte count,
// and decode is bandwidth-bound (TPOT = bytes / BW), so bytes set the time.
// Only cross-step expert caching (hot experts pinned across steps, routing-
// locality dependent) could go below this; v1 assumes a full re-read per step.
export function expertCoverage(experts: number, expertsPerToken: number, batchSize: number): number {
  return 1 - Math.pow(1 - expertsPerToken / experts, batchSize);
}

export interface DecodeDetail {
  expertCoverage: number; // 1 for dense
  weightsReadBytes: number; // per GPU per step
  kvReadBytes: number; // per GPU per step
  bytesPerStep: number; // per GPU
  sHistory: number; // history length used
  tBandwidthMs: number;
  tCommMs: number;
  tpotMs: number;
}

export function decodeStepTime(inp: PhaseInput, sHistoryOverride?: number): DecodeDetail {
  const { model, derived, gpu, layout, workload, cal, comm } = inp;
  const B = workload.batchSize;
  // Average history length across the N_out generation steps
  // (history grows from N_in + 1 to N_in + N_out).
  const sHistory = sHistoryOverride ?? workload.inputLen + (workload.outputLen + 1) / 2;
  const sf = shardFactors(layout);

  let coverage = 1;
  let weightsReadBytes: number;
  if (model.type === 'moe' && model.moe) {
    coverage = expertCoverage(model.moe.experts, model.moe.expertsPerToken, B);
    weightsReadBytes =
      derived.wNonexpertBytes / sf.weightNonExpert +
      (coverage * derived.wExpertBytes) / sf.weightExpert;
  } else {
    weightsReadBytes = derived.wBytesTotal / sf.weightNonExpert;
  }

  const kvReadBytes = (derived.kv.totalBytes(sHistory) * B) / sf.kvAndActivation;
  const bytesPerStep = weightsReadBytes + kvReadBytes;
  const tBandwidthMs = (bytesPerStep / (gpu.bwGbps * 1e9 * cal.bwEffDecode)) * 1e3;

  // TP comm per step: decode advances one token per sequence, msg = B * h * b_act.
  const msgBytes = B * model.hiddenSize * B_ACT;
  const tTpMs = model.layers * 2 * ringAllreduceMs(msgBytes, layout.tp, comm);

  let tEpMs = 0;
  if (model.type === 'moe' && model.moe && layout.ep > 1) {
    const buffer = B * model.moe.expertsPerToken * model.hiddenSize * B_ACT;
    const sendBytes = (buffer * (layout.ep - 1)) / layout.ep;
    tEpMs = model.layers * 2 * allToAllMs(sendBytes, layout.ep, comm);
  }

  const tCommMs = tTpMs + tEpMs;
  const exposedTpMs = exposedComm(tTpMs, cal.tpCommOverlap);
  const exposedEpMs = exposedComm(tEpMs, cal.epCommOverlap);
  const tpotMs = tBandwidthMs + exposedTpMs + exposedEpMs;

  return {
    expertCoverage: coverage,
    weightsReadBytes,
    kvReadBytes,
    bytesPerStep,
    sHistory,
    tBandwidthMs,
    tCommMs,
    tpotMs,
  };
}

// Speculative decoding decode step details.
export interface SdDecodeDetail {
  draftStep: DecodeDetail; // one draft forward pass
  verifyStep: DecodeDetail; // one verification forward pass
  gamma: number;  // num tokens
  acceptanceRate: number;
  expectedTokensPerCycle: number; // γ·α + 1
  cycleTimeMs: number; // γ · draftStep.tpotMs + verifyStep.tpotMs
  tpotMs: number; // cycleTimeMs / expectedTokensPerCycle
  baselineTpotMs: number; // standard TPOT without SD
  speedup: number; // baselineTpotMs / tpotMs
}

// Compute speculative decoding decode step time.
// Draft model generates γ tokens, then main model verifies in one pass.
// Both models use the same GPU (same HBM BW, same peak FLOPS).
export function sdDecodeStepTime(
  mainInp: PhaseInput,
  draftInp: PhaseInput,
  gamma: number,
  acceptanceRate: number,
): SdDecodeDetail {
  // Draft step: use draft model parameters and layout
  const draftStep = decodeStepTime(draftInp);

  // Verify step: use main model
  // For v1, ignore the small KV cache growth correction (γ << seqLen)
  const verifyStep = decodeStepTime(mainInp);

  // Baseline: standard TPOT without SD
  const baselineTpotMs = verifyStep.tpotMs;

  // SD formula
  const expectedTokensPerCycle = gamma * acceptanceRate + 1;
  const cycleTimeMs = gamma * draftStep.tpotMs + verifyStep.tpotMs;
  const tpotMs = cycleTimeMs / expectedTokensPerCycle;
  const speedup = baselineTpotMs / tpotMs;

  return {
    draftStep,
    verifyStep,
    gamma,
    acceptanceRate,
    expectedTokensPerCycle,
    cycleTimeMs,
    tpotMs,
    baselineTpotMs,
    speedup,
  };
}
