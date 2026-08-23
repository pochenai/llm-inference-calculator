// Tests for the automatic parallel-layout solver.

import { describe, expect, it } from 'vitest';
import type { ModelSpec } from '../src/core/types.js';
import { solveParallelLayout } from '../src/core/solver.js';
import type { SolverInput } from '../src/core/solver.js';
import { model } from '../src/data/models/index.js';
import { gpu } from '../src/data/gpus/nvidia/index.js';

// Fixtures come straight from the bundled data catalog (src/data).
const llama70b = model('llama3_1_70b');
const qwen3Moe = model('qwen3_235b');
const h100 = gpu('h100_sxm');

function baseInput(over?: Partial<SolverInput>): SolverInput {
  return {
    model: llama70b,
    gpu: h100,
    gpusPerNode: 8,
    numGpus: 8,
    dp: 1,
    ep: 1,
    intraNodeBwGbps: 900,
    interNodeBwGbps: 50,
    workload: { batchSize: 8, inputLen: 2048, outputLen: 512 },
    weightQuant: 'fp16',
    kvQuant: 'fp16',
    flashAttention: true,
    headroom: 0.1,
    ...over,
  };
}

describe('solveParallelLayout', () => {
  it('picks the largest feasible TP for Llama 70B fp16 on 8xH100', () => {
    const r = solveParallelLayout(baseInput());
    expect(r.issues).toEqual([]);
    expect(r.chosen).toEqual({ tp: 8, pp: 1, ep: 1, dp: 1 });
    // Smaller-TP layouts (TP=4 PP=2, ...) fit too and are offered as options.
    expect(r.feasibleLayouts.length).toBeGreaterThanOrEqual(1);
    expect(r.feasibleLayouts[0]?.tp).toBe(8);
  });

  it('returns bestEffort (largest TP) when nothing fits at all', () => {
    // 400B fp16 = 800 GB weights > 8 x 72 GB usable, so no layout fits.
    const big: ModelSpec = { ...llama70b, paramsB: 400 };
    const r = solveParallelLayout(baseInput({ model: big, numGpus: 8 }));
    expect(r.chosen).toBeNull();
    expect(r.bestEffort).toEqual({ tp: 8, pp: 1, ep: 1, dp: 1 });
  });

  it('uses PP across nodes when TP hits the gpusPerNode cap', () => {
    // 16 GPUs, 8 per node. 400B fp16 needs >= 10 shards of 80 GB capacity,
    // so the solver must combine TP=8 with PP=2.
    const big: ModelSpec = { ...llama70b, paramsB: 400 };
    const r = solveParallelLayout(baseInput({ model: big, numGpus: 16 }));
    expect(r.chosen).toEqual({ tp: 8, pp: 2, ep: 1, dp: 1 });
  });

  it('sanitizes DP that does not divide the GPU count', () => {
    const r = solveParallelLayout(baseInput({ dp: 3 }));
    expect(r.dp).toBe(1);
    expect(r.issues.some((s) => s.includes('DP=3'))).toBe(true);
    expect(r.chosen).toEqual({ tp: 8, pp: 1, ep: 1, dp: 1 });
  });

  it('rejects EP on dense models and clamps EP to a divisor', () => {
    // Dense model: EP is forced to 1 with a note.
    const dense = solveParallelLayout(baseInput({ ep: 4 }));
    expect(dense.ep).toBe(1);
    expect(dense.issues.some((s) => s.includes('requires a MoE model'))).toBe(true);

    // MoE, 8 GPUs: EP=3 does not divide 8 -> clamp to the largest divisor <= 3.
    const moe = solveParallelLayout(baseInput({ model: qwen3Moe, ep: 3, numGpus: 8 }));
    expect(moe.ep).toBe(2);
    expect(moe.issues.some((s) => s.includes('EP=3'))).toBe(true);
  });

  it('keeps user EP for MoE when it divides and shards expert weights', () => {
    // Qwen3 235B fp16 = 470 GB total. On 8xH100 (72 GB usable each = 576 GB)
    // it only fits when experts are sharded: TP=2 x EP=4 divides weights by 8.
    const r = solveParallelLayout(
      baseInput({ model: qwen3Moe, ep: 4, weightQuant: 'fp16', kvQuant: 'fp16' }),
    );
    expect(r.ep).toBe(4);
    expect(r.chosen).not.toBeNull();
    expect(r.chosen!.tp * r.chosen!.pp * r.chosen!.ep * r.chosen!.dp).toBe(8);
  });

  it('respects DP partitioning when DP divides the GPU count', () => {
    // 8 GPUs with DP=2 => 4 GPUs per replica, TP capped at 4.
    const small: ModelSpec = { ...llama70b, paramsB: 7, layers: 32 };
    const r = solveParallelLayout(baseInput({ model: small, dp: 2 }));
    expect(r.dp).toBe(2);
    expect(r.chosen).toEqual({ tp: 4, pp: 1, ep: 1, dp: 2 });
  });

  it('reports infeasible with bestEffort for an oversized model on one GPU', () => {
    const r = solveParallelLayout(baseInput({ numGpus: 1 }));
    expect(r.chosen).toBeNull();
    expect(r.bestEffort).toEqual({ tp: 1, pp: 1, ep: 1, dp: 1 });
    expect(r.feasibleLayouts).toEqual([]);
  });

  it('produces layouts that pass validateLayout (product = N_gpu, TP <= gpusPerNode)', () => {
    const r = solveParallelLayout(baseInput({ model: qwen3Moe, ep: 2, numGpus: 16, dp: 2 }));
    for (const l of r.feasibleLayouts) {
      expect(l.tp * l.ep * l.pp * l.dp).toBe(16);
      expect(l.tp).toBeLessThanOrEqual(8);
    }
  });
});
