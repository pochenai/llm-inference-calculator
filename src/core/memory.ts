// VRAM model: per-GPU breakdown, feasibility check, and B_max inversion.

import type { GpuSpec, ModelSpec, ParallelLayout, Workload } from './types';
import { B_ACT, GiB } from './types';
import type { DerivedConstants } from './model';
import { qDimOf } from './model';
import { shardFactors } from './layout';

export interface MemoryOptions {
  flashAttention: boolean;
  headroom: number; // reserved capacity fraction, e.g. 0.1
  overheadBytes?: number; // CUDA context, allocator fragmentation, ...
}

export interface VramBreakdown {
  weightsBytes: number; // per GPU
  kvBytes: number; // per GPU, at full workload length
  activationBytes: number; // per GPU
  overheadBytes: number;
  totalBytes: number;
  capacityBytes: number; // vram * (1 - headroom)
  feasible: boolean;
  bMax: number; // max batch under the VRAM constraint
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
): VramBreakdown {
  const sf = shardFactors(layout);
  const overheadBytes = opts.overheadBytes ?? GiB;
  const capacityBytes = gpu.vramGb * 1e9 * (1 - opts.headroom);

  const seqLen = workload.inputLen + workload.outputLen;
  const kvPerSeq = derived.kv.totalBytes(seqLen) / sf.kvAndActivation;
  const actPerSeq =
    activationBytesPerSeq(model, derived, workload, opts.flashAttention) / sf.kvAndActivation;
  const weights = weightsPerGpu(derived, model, layout);

  const totalBytes = weights + (kvPerSeq + actPerSeq) * workload.batchSize + overheadBytes;
  const feasible = totalBytes <= capacityBytes;

  let bMax: number = 0;
  const budget = capacityBytes - weights - overheadBytes;
  if (budget > 0 && kvPerSeq + actPerSeq > 0) {
    bMax = Math.floor(budget / (kvPerSeq + actPerSeq));
  }

  return {
    weightsBytes: weights,
    kvBytes: kvPerSeq * workload.batchSize,
    activationBytes: actPerSeq * workload.batchSize,
    overheadBytes,
    totalBytes,
    capacityBytes,
    feasible,
    bMax,
  };
}
