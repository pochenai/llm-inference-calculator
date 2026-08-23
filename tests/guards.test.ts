// Tests for defensive guards: contradictory inputs must throw with informative messages.

import { describe, expect, it } from 'vitest';
import type { GpuSpec, ModelSpec, SystemSpec } from '../src/core/types.js';
import {
  buildKvGeometry,
  attentionQuadFlops,
  nonExpertParamsB,
  deriveConstants,
} from '../src/core/model.js';
import { BYTES_PER_PARAM } from '../src/core/types.js';
import { evaluate } from '../src/core/metrics.js';

const bKv = BYTES_PER_PARAM.fp16;

// Baseline valid model for mutation.
const validDense: ModelSpec = {
  id: 'test_dense',
  name: 'Test Dense',
  type: 'dense',
  paramsB: 7,
  layers: 32,
  hiddenSize: 4096,
  kvHeads: 8,
  headDim: 128,
  maxCtx: 8192,
};

const validMoe: ModelSpec = {
  id: 'test_moe',
  name: 'Test MoE',
  type: 'moe',
  paramsB: 235,
  layers: 94,
  hiddenSize: 4096,
  kvHeads: 4,
  headDim: 128,
  maxCtx: 262144,
  moe: { experts: 128, expertsPerToken: 8, activeParamsB: 22, execution: 'shared_routed' },
};

describe('buildKvGeometry guard: contradictory layer counts', () => {
  it('throws when special layers exceed total layers', () => {
    // linearAttentionLayers=20 + localLayers=20 = 40, but layers=32
    const bad: ModelSpec = {
      ...validDense,
      layers: 32,
      linearAttentionLayers: 20,
      localLayers: 20,
      slidingWindow: 512,
    };
    expect(() => buildKvGeometry(bad, bKv)).toThrow(/layers/i);
    expect(() => buildKvGeometry(bad, bKv)).toThrow(/contradictory/i);
  });

  it('throws when mambaRatio-derived SSM layers plus local exceed total', () => {
    // mambaRatio=0.8 -> ssmLayers = round(0.8*32) = 26, localLayers=10 -> 36 > 32
    const bad: ModelSpec = {
      ...validDense,
      layers: 32,
      mambaRatio: 0.8,
      localLayers: 10,
      slidingWindow: 256,
    };
    expect(() => buildKvGeometry(bad, bKv)).toThrow(/layers/i);
  });

  it('includes model id and field values in the error message', () => {
    const bad: ModelSpec = {
      ...validDense,
      id: 'my_bad_model',
      layers: 10,
      linearAttentionLayers: 8,
      localLayers: 5,
      slidingWindow: 128,
    };
    expect(() => buildKvGeometry(bad, bKv)).toThrow(/my_bad_model/);
    expect(() => buildKvGeometry(bad, bKv)).toThrow(/layers=10/);
  });

  it('does NOT throw when special layers exactly equal total', () => {
    // globalLayers = 32 - 32 = 0, which is fine (all layers are special).
    const ok: ModelSpec = {
      ...validDense,
      layers: 32,
      linearAttentionLayers: 32,
    };
    expect(() => buildKvGeometry(ok, bKv)).not.toThrow();
  });
});

describe('totalBytes guard: negative seqLen', () => {
  it('throws on negative seqLen', () => {
    const kv = buildKvGeometry(validDense, bKv);
    expect(() => kv.totalBytes(-1)).toThrow(/seqLen=-1/);
    expect(() => kv.totalBytes(-100)).toThrow(/negative/i);
  });

  it('accepts seqLen=0', () => {
    const kv = buildKvGeometry(validDense, bKv);
    expect(() => kv.totalBytes(0)).not.toThrow();
    expect(kv.totalBytes(0)).toBe(0);
  });
});

describe('attentionQuadFlops guard: negative seqLen', () => {
  it('throws on negative seqLen', () => {
    const kv = buildKvGeometry(validDense, bKv);
    expect(() => attentionQuadFlops(validDense, kv, -5)).toThrow(/seqLen=-5/);
    expect(() => attentionQuadFlops(validDense, kv, -1)).toThrow(/negative/i);
  });

  it('accepts seqLen=0', () => {
    const kv = buildKvGeometry(validDense, bKv);
    expect(() => attentionQuadFlops(validDense, kv, 0)).not.toThrow();
    expect(attentionQuadFlops(validDense, kv, 0)).toBe(0);
  });
});

describe('nonExpertParamsB guard: inconsistent MoE data', () => {
  it('throws when solved nonExpertParams is negative', () => {
    // N = (paramsB*k - experts*activeParamsB) / (k - experts)
    // With paramsB=10, k=2, experts=4, activeParamsB=1:
    // N = (10*2 - 4*1) / (2 - 4) = 16 / (-2) = -8 -> negative, throw.
    const bad: ModelSpec = {
      ...validDense,
      type: 'moe',
      paramsB: 10,
      moe: { experts: 4, expertsPerToken: 2, activeParamsB: 1, execution: 'shared_routed' },
    };
    expect(() => nonExpertParamsB(bad)).toThrow(/inconsistent/i);
    expect(() => nonExpertParamsB(bad)).toThrow(/test_moe|test_dense/);
  });

  it('throws when solved nonExpertParams exceeds activeParamsB', () => {
    // N = (paramsB*k - experts*activeParamsB) / (k - experts)
    // paramsB=100, k=2, experts=4, activeParamsB=10:
    // N = (100*2 - 4*10) / (2 - 4) = 160 / (-2) = -80 -> negative, throw.
    // Try paramsB=5, k=3, experts=4, activeParamsB=10:
    // N = (5*3 - 4*10) / (3 - 4) = (-25) / (-1) = 25 -> 25 > activeParamsB=10, throw.
    const bad: ModelSpec = {
      ...validDense,
      type: 'moe',
      paramsB: 5,
      moe: { experts: 4, expertsPerToken: 3, activeParamsB: 10, execution: 'shared_routed' },
    };
    expect(() => nonExpertParamsB(bad)).toThrow(/inconsistent/i);
    expect(() => nonExpertParamsB(bad)).toThrow(/nonExpertParamsB=25/);
  });

  it('includes model id and all relevant field values in the error', () => {
    const bad: ModelSpec = {
      ...validDense,
      id: 'broken_moe',
      type: 'moe',
      paramsB: 10,
      moe: { experts: 4, expertsPerToken: 2, activeParamsB: 1, execution: 'shared_routed' },
    };
    expect(() => nonExpertParamsB(bad)).toThrow(/broken_moe/);
    expect(() => nonExpertParamsB(bad)).toThrow(/paramsB=10/);
    expect(() => nonExpertParamsB(bad)).toThrow(/activeParamsB=1/);
    expect(() => nonExpertParamsB(bad)).toThrow(/experts=4/);
    expect(() => nonExpertParamsB(bad)).toThrow(/expertsPerToken=2/);
  });

  it('does NOT throw for consistent MoE data (Qwen3 235B)', () => {
    // N = (235*8 - 128*22) / (8 - 128) = 7.8, valid.
    expect(() => nonExpertParamsB(validMoe)).not.toThrow();
    expect(nonExpertParamsB(validMoe)).toBeCloseTo(7.8, 6);
  });

  it('does NOT throw for dense models (bypasses MoE path)', () => {
    expect(() => nonExpertParamsB(validDense)).not.toThrow();
    expect(nonExpertParamsB(validDense)).toBe(validDense.paramsB);
  });
});

describe('valid fixtures remain unaffected by new guards', () => {
  it('deriveConstants works for a valid dense model', () => {
    expect(() => deriveConstants(validDense, 'fp16', 'fp16')).not.toThrow();
  });

  it('deriveConstants works for a valid MoE model', () => {
    expect(() => deriveConstants(validMoe, 'fp16', 'fp16')).not.toThrow();
  });
});

// --- UI contract: evaluate() returns typed errors instead of throwing ---

const h100: GpuSpec = {
  id: 'h100_sxm',
  name: 'H100 SXM5',
  vramGb: 80,
  bwGbps: 3350,
  peakTflops: { bf16: 989, fp8: 1979, int8: 1979, int4: 3958 },
  nvlinkBwGbps: 900,
};

function specWith(model: ModelSpec): SystemSpec {
  return {
    model,
    gpu: h100,
    gpusPerNode: 8,
    interNodeBwGbps: 50,
    workload: { batchSize: 1, inputLen: 512, outputLen: 64 },
    weightQuant: 'fp16',
    kvQuant: 'fp16',
    layout: { tp: 8, pp: 1, ep: 1, dp: 1 },
    flashAttention: true,
    headroom: 0.1,
  };
}

describe('evaluate returns typed errors instead of throwing (UI contract)', () => {
  it('returns contradictory-layers for impossible layer counts', () => {
    const bad: ModelSpec = {
      ...validDense,
      layers: 10,
      linearAttentionLayers: 8,
      localLayers: 5,
      slidingWindow: 128,
    };
    const r = evaluate(specWith(bad));
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error.code).toBe('contradictory-layers');
  });

  it('returns inconsistent-moe for contradictory MoE params', () => {
    const bad: ModelSpec = {
      ...validDense,
      type: 'moe',
      paramsB: 10,
      moe: { experts: 4, expertsPerToken: 2, activeParamsB: 1, execution: 'shared_routed' },
    };
    const r = evaluate(specWith(bad));
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error.code).toBe('inconsistent-moe');
  });

  it('returns invalid-layout when TP exceeds gpus-per-node', () => {
    // Non-disagg total GPU count is derived from the layout product, so the
    // reachable invalid-layout cases are structural (EP on dense, TP > node).
    const spec = specWith(validDense);
    spec.layout = { tp: 16, pp: 1, ep: 1, dp: 1 }; // tp=16 > gpusPerNode=8
    const r = evaluate(spec);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error.code).toBe('invalid-layout');
  });

  it('returns invalid-layout when EP is used on a dense model', () => {
    const spec = specWith(validDense);
    spec.layout = { tp: 4, pp: 1, ep: 2, dp: 1 }; // ep=2 but model is dense
    const r = evaluate(spec);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error.code).toBe('invalid-layout');
  });

  it('returns negative-seqlen for negative input length', () => {
    const spec = specWith(validDense);
    spec.workload = { ...spec.workload, inputLen: -5 };
    const r = evaluate(spec);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error.code).toBe('negative-seqlen');
  });

  it('returns ok:true for a valid spec', () => {
    const r = evaluate(specWith(validDense));
    expect(r.ok).toBe(true);
  });
});
