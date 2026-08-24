// VRAM model: per-GPU breakdown, feasibility check, and B_max inversion.

import type { GpuSpec, ModelSpec, ParallelLayout, Workload } from './types';
import { B_ACT, GiB } from './types';
import type { DerivedConstants } from './model';
import { qDimOf } from './model';
import { shardFactors } from './layout';

// PD disaggregation modes for VRAM sizing.
// 'prefill': only inputLen tokens (no output generated on prefill GPUs).
// 'decode': inputLen + outputLen/2 (steady-state avg under continuous batching).
export const PD_PREFILL = 'prefill' as const;
export const PD_DECODE = 'decode' as const;
export type PdMode = typeof PD_PREFILL | typeof PD_DECODE;

export interface MemoryOptions {
  flashAttention: boolean;
  headroom: number; // reserved capacity fraction, e.g. 0.1
  overheadBytes?: number; // CUDA context, allocator fragmentation, ...
  pdMode?: PdMode;
}

export interface VramBreakdown {
  weightsBytes: number; // per GPU (main weights only, for backward compat)
  kvBytes: number; // per GPU, at full workload length
  activationBytes: number; // per GPU
  overheadBytes: number;
  totalBytes: number;
  capacityBytes: number; // vram * (1 - headroom)
  feasible: boolean;
  bMax: number; // max batch under the VRAM constraint
  // Speculative decoding components (present when SD enabled)
  draftWeightsBytes?: number; // draft model weights per GPU
  draftKvBytes?: number; // draft model KV per GPU
}

// Per-sequence activation bytes (scales linearly with B in both FA modes).
export function activationBytesPerSeq(
  model: ModelSpec,
  derived: DerivedConstants,
  workload: Workload,
  flashAttention: boolean,
): number {
  // Ideal model: approximate live activations as a single O(N*h) residual buffer.
  // Real runtimes hold more buffers; this is a documented simplification, and
  // the term is small next to weights + KV anyway.
  const base = workload.inputLen * model.hiddenSize * B_ACT;
  if (flashAttention) return base;
  // Without FlashAttention the attention score matrix [q_heads, N, N] is
  // materialized at the single live layer: q_heads * N^2 elements per sequence.
  // (The batch factor B is applied by the caller in vramBreakdown.)
  const scoreMatrix = derived.qHeads * workload.inputLen * workload.inputLen * B_ACT;
  return base + scoreMatrix;
}

export function weightsPerGpu(derived: DerivedConstants, model: ModelSpec, layout: ParallelLayout): number {
  const sf = shardFactors(layout);
  if (model.type === 'moe') {
    return derived.wNonexpertBytes / sf.weightNonExpert + derived.wExpertBytes / sf.weightExpert;
  }
  return derived.wBytesTotal / sf.weightNonExpert;
}

export function vramBreakdown(
  model: ModelSpec,
  derived: DerivedConstants,
  gpu: GpuSpec,
  layout: ParallelLayout,
  workload: Workload,
  opts: MemoryOptions,
  // Optional speculative decoding parameters
  draftModel?: ModelSpec,
  draftDerived?: DerivedConstants,
  draftTp?: number,
): VramBreakdown {
  const sf = shardFactors(layout);
  const overheadBytes = opts.overheadBytes ?? GiB;
  const capacityBytes = gpu.vramGb * 1e9 * (1 - opts.headroom);

  // PD disaggregation adjusts effective sequence length for KV cache sizing:
  // - prefill: only inputLen (output tokens never live on prefill GPUs)
  // - decode: inputLen + outputLen/2 (steady-state avg under continuous batching)
  let seqLen: number;
  if (opts.pdMode === PD_PREFILL) {
    seqLen = workload.inputLen;
  } else if (opts.pdMode === PD_DECODE) {
    seqLen = workload.inputLen + Math.ceil(workload.outputLen / 2);
  } else {
    seqLen = workload.inputLen + workload.outputLen;
  }
  const mainKvPerSeq = derived.kv.totalBytes(seqLen) / sf.kvAndActivation;
  const mainActPerSeq =
    activationBytesPerSeq(model, derived, workload, opts.flashAttention) / sf.kvAndActivation;
  const mainWeights = weightsPerGpu(derived, model, layout);

  // Speculative decoding: account for draft model in VRAM
  let draftWeights: number | undefined;
  let draftKvPerSeq: number | undefined;
  let draftActPerSeq: number | undefined;
  let totalWeights = mainWeights;
  let kvPerSeqTotal = mainKvPerSeq;
  let actPerSeq = mainActPerSeq;

  if (draftModel && draftDerived && draftTp) {
    const draftLayout: ParallelLayout = { tp: draftTp, pp: 1, ep: 1, dp: 1 };
    draftWeights = weightsPerGpu(draftDerived, draftModel, draftLayout);
    draftKvPerSeq = draftDerived.kv.totalBytes(seqLen) / draftTp;
    draftActPerSeq =
      activationBytesPerSeq(draftModel, draftDerived, workload, opts.flashAttention) / draftTp;

    totalWeights = mainWeights + draftWeights;
    kvPerSeqTotal = mainKvPerSeq + draftKvPerSeq;
    // Activations: take max (draft and main don't run simultaneously)
    actPerSeq = Math.max(mainActPerSeq, draftActPerSeq);
  }

  // With DP, each replica handles batchSize/dp sequences
  const batchPerReplica = workload.batchSize / layout.dp;
  const totalBytes = totalWeights + (kvPerSeqTotal + actPerSeq) * batchPerReplica + overheadBytes;
  const feasible = totalBytes <= capacityBytes;

  let bMax: number = 0;
  const budget = capacityBytes - totalWeights - overheadBytes;
  if (budget > 0 && kvPerSeqTotal + actPerSeq > 0) {
    // bMax is per-GPU, so it's per-replica max batch
    bMax = Math.floor(budget / (kvPerSeqTotal + actPerSeq));
  }

  const result: VramBreakdown = {
    weightsBytes: mainWeights,
    kvBytes: mainKvPerSeq * batchPerReplica,
    activationBytes: actPerSeq * batchPerReplica,
    overheadBytes,
    totalBytes,
    capacityBytes,
    feasible,
    bMax,
  };

  // Add draft components if SD enabled
  if (draftWeights !== undefined && draftKvPerSeq !== undefined) {
    result.draftWeightsBytes = draftWeights;
    result.draftKvBytes = draftKvPerSeq * batchPerReplica;
  }

  return result;
}
