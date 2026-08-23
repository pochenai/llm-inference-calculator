// Parallel layout: constraint validation and per-GPU shard factors.

import type { ModelSpec, ParallelLayout } from './types.js';

// Validate TP * EP * PP * DP = N_gpu and structural rules.
// Returns a list of error messages (empty = valid).
export function validateLayout(
  layout: ParallelLayout,
  model: ModelSpec,
  numGpus: number,
  gpusPerNode: number,
): string[] {
  const errors: string[] = [];
  const { tp, ep, pp, dp } = layout;

  for (const [name, v] of Object.entries({ tp, ep, pp, dp })) {
    if (!Number.isInteger(v) || v < 1) errors.push(`${name} must be an integer >= 1 (got ${v})`);
  }
  if (errors.length > 0) return errors;

  const product = tp * ep * pp * dp;
  if (product !== numGpus) {
    errors.push(`TP*EP*PP*DP = ${product} but N_gpu = ${numGpus}`);
  }
  if (ep > 1 && model.type !== 'moe') {
    errors.push(`EP = ${ep} requires a MoE model ('${model.id}' is dense)`);
  }
  if (tp > gpusPerNode) {
    errors.push(`TP = ${tp} exceeds gpusPerNode = ${gpusPerNode} (TP must fit within one node)`);
  }
  return errors;
}

export interface ShardFactors {
  // Divisor for non-expert weights (attention, shared experts, embeddings).
  weightNonExpert: number;
  // Divisor for routed expert weights (EP distributes whole experts,
  // TP further shards within the EP group). Never apply TP and EP to the
  // same weights twice beyond this product.
  weightExpert: number;
  // Divisor for KV cache and activations (attention-side, TP/PP only).
  kvAndActivation: number;
}

export function shardFactors(layout: ParallelLayout): ShardFactors {
  return {
    weightNonExpert: layout.tp * layout.pp,
    weightExpert: layout.tp * layout.ep * layout.pp,
    kvAndActivation: layout.tp * layout.pp,
  };
}
