// Sanity tests for the calculator core (ideal-value mode).

import { describe, expect, it } from 'vitest';
import type { GpuSpec, ModelSpec, SystemSpec } from '../src/core/types.js';
import { deriveConstants, nonExpertParamsB, buildKvGeometry, attentionQuadFlops } from '../src/core/model.js';
import { activationBytesPerSeq } from '../src/core/memory.js';
import { validateLayout } from '../src/core/layout.js';
import { expertCoverage, ringAllreduceMs } from '../src/core/latency.js';
import { evaluate as evaluateResult } from '../src/core/metrics.js';
import type { EvaluationResult } from '../src/core/metrics.js';
import { BYTES_PER_PARAM } from '../src/core/types.js';
import { IDEAL } from '../src/core/calibration.js';
import type { Calibration } from '../src/core/calibration.js';
import { unwrap } from '../src/core/errors.js';

// evaluate() now returns Result<EvaluationResult> so a UI can branch on r.ok.
// Success-path tests unwrap it (throws if the calculator unexpectedly errors).
function evaluate(spec: SystemSpec, cal?: Calibration): EvaluationResult {
  return unwrap(evaluateResult(spec, cal));
}

const llama70b: ModelSpec = {
  id: 'llama3_1_70b',
  name: 'Llama 3.1 70B',
  type: 'dense',
  paramsB: 70.6,
  layers: 80,
  hiddenSize: 8192,
  kvHeads: 8,
  headDim: 128,
  maxCtx: 131072,
};

const qwen3Moe: ModelSpec = {
  id: 'qwen3_235b',
  name: 'Qwen3 235B A22B',
  type: 'moe',
  paramsB: 235,
  layers: 94,
  hiddenSize: 4096,
  kvHeads: 4,
  headDim: 128,
  maxCtx: 262144,
  moe: { experts: 128, expertsPerToken: 8, activeParamsB: 22, execution: 'shared_routed' },
};

const gemma3Like: ModelSpec = {
  id: 'gemma3_27b_like',
  name: 'Gemma 3 27B (sliding window)',
  type: 'dense',
  paramsB: 27,
  layers: 62,
  hiddenSize: 5376,
  kvHeads: 16,
  headDim: 128,
  localLayers: 52,
  slidingWindow: 1024,
  maxCtx: 131072,
};

const h100: GpuSpec = {
  id: 'h100_sxm',
  name: 'H100 SXM5',
  vramGb: 80,
  bwGbps: 3350,
  peakTflops: { bf16: 989, fp8: 1979, int8: 1979, int4: 3958 },
  nvlinkBwGbps: 900,
};

function llama8xH100(over?: Partial<SystemSpec>): SystemSpec {
  return {
    model: llama70b,
    gpu: h100,
    gpusPerNode: 8,
    interNodeBwGbps: 50,
    workload: { batchSize: 8, inputLen: 2048, outputLen: 512 },
    weightQuant: 'fp16',
    kvQuant: 'fp16',
    layout: { tp: 8, pp: 1, ep: 1, dp: 1 },
    flashAttention: true,
    headroom: 0.1,
    ...over,
  };
}

describe('model derived constants', () => {
  it('KV per token for Llama 3.1 70B (fp16 KV) is 320 KB', () => {
    const kv = buildKvGeometry(llama70b, BYTES_PER_PARAM.fp16);
    // 2 (K+V) * 2 bytes * 80 layers * 8 kv heads * 128 head_dim = 327,680
    expect(kv.bytesPerToken).toBe(327680);
    expect(kv.totalBytes(1000)).toBe(327680 * 1000);
    expect(kv.globalLayers).toBe(80);
    expect(kv.localLayers).toBe(0);
  });

  it('weight bytes respect quantization', () => {
    const d16 = deriveConstants(llama70b, 'fp16', 'fp16');
    expect(d16.wBytesTotal).toBeCloseTo(70.6e9 * 2, -3);
    const d8 = deriveConstants(llama70b, 'fp8', 'fp16');
    expect(d8.wBytesTotal).toBeCloseTo(70.6e9, -3);
  });

  it('solves MoE non-expert params for Qwen3 235B', () => {
    // N = (P*k - E*A) / (k - E) = (235*8 - 128*22) / (8 - 128) = 7.8
    expect(nonExpertParamsB(qwen3Moe)).toBeCloseTo(7.8, 6);
  });

  it('sliding-window layers cap KV at the window', () => {
    const kv = buildKvGeometry(gemma3Like, BYTES_PER_PARAM.fp16);
    expect(kv.globalLayers).toBe(10);
    expect(kv.localLayers).toBe(52);
    const elemsPerLayer = 16 * 128; // kv_heads * head_dim
    const beyondWindow = 100000;
    const expected =
      2 * 2 * (10 * elemsPerLayer * beyondWindow + 52 * elemsPerLayer * 1024);
    expect(kv.totalBytes(beyondWindow)).toBe(expected);
    // bytesPerToken is the small-S marginal cost (all layers still filling);
    // sub-linearity shows up in totalBytes capping local layers at the window.
    expect(kv.bytesPerToken).toBe(2 * 2 * (10 + 52) * elemsPerLayer);
  });

  it('attention quadratic FLOPs scale as 4 * N^2 * qDim per full layer', () => {
    const kv = buildKvGeometry(llama70b, 2);
    const n = 2048;
    const flops = attentionQuadFlops(llama70b, kv, n);
    const qDim = (8192 / 128) * 128; // q_heads ~= hidden / head_dim
    expect(flops).toBe(4 * qDim * 80 * n * n);
  });
});

describe('layout validation', () => {
  it('rejects product mismatch, EP on dense, and cross-node TP', () => {
    expect(validateLayout({ tp: 4, pp: 1, ep: 1, dp: 1 }, llama70b, 8, 8)).toEqual([
      'TP*EP*PP*DP = 4 but N_gpu = 8',
    ]);
    expect(validateLayout({ tp: 8, pp: 1, ep: 2, dp: 1 }, llama70b, 16, 8)[0]).toContain(
      'requires a MoE model',
    );
    expect(validateLayout({ tp: 16, pp: 1, ep: 1, dp: 1 }, llama70b, 16, 8)[0]).toContain(
      'exceeds gpusPerNode',
    );
    expect(validateLayout({ tp: 8, pp: 1, ep: 1, dp: 1 }, llama70b, 8, 8)).toEqual([]);
  });
});

describe('expert coverage', () => {
  it('covers k/E of experts at batch 1 and saturates at large batch', () => {
    expect(expertCoverage(128, 8, 1)).toBeCloseTo(8 / 128, 10);
    expect(expertCoverage(128, 8, 10000)).toBeGreaterThan(0.999);
  });
});

describe('VRAM model', () => {
  it('fits Llama 70B fp16 on 8xH100 TP=8', () => {
    const r = evaluate(llama8xH100());
    expect(r.feasible).toBe(true);
    // weights per GPU = 141.2 GB / 8 = 17.65 GB
    expect(r.memory.weightsBytes).toBeCloseTo((70.6e9 * 2) / 8, -6);
  });

  it('reports infeasible on a single H100 with B_max = 0', () => {
    const r = evaluate(llama8xH100({ layout: { tp: 1, pp: 1, ep: 1, dp: 1 } }));
    expect(r.feasible).toBe(false);
    expect(r.memory.bMax).toBe(0);
  });

  it('inverts B_max on a single A100-sized budget', () => {
    // Shrink the model so weights fit; B_max should be KV-limited and positive.
    const small: ModelSpec = { ...llama70b, paramsB: 7, layers: 32 };
    const r = evaluate(
      llama8xH100({ model: small, layout: { tp: 1, pp: 1, ep: 1, dp: 1 } }),
    );
    expect(r.feasible).toBe(true);
    expect(r.memory.bMax).toBeGreaterThan(8);
  });

  it('no-FA activation adds q_heads*N^2 score matrix (not qDim*N^2)', () => {
    const derived = deriveConstants(llama70b, 'fp16', 'fp16');
    const wl = { batchSize: 1, inputLen: 1024, outputLen: 0 };
    const withFA = activationBytesPerSeq(llama70b, derived, wl, true);
    const noFA = activationBytesPerSeq(llama70b, derived, wl, false);
    // llama70b q_heads = 8192/128 = 64. The [q_heads, N, N] score matrix adds
    // q_heads*N^2*b_act; using qDim (= q_heads*head_dim) would overstate 128x.
    expect(noFA - withFA).toBe(64 * 1024 * 1024 * 2);
  });
});

describe('latency model (ideal values)', () => {
  it('TTFT lands in the expected roofline range for 8xH100', () => {
    const r = evaluate(llama8xH100({ workload: { batchSize: 1, inputLen: 2048, outputLen: 512 } }));
    // FLOPs ~= 2048 * 2 * 70.6e9 (+ ~1.1e13 attention) ~= 3.0e14
    // 8 * 989 TFLOPS -> ~38 ms
    expect(r.ttftMs).toBeGreaterThan(30);
    expect(r.ttftMs).toBeLessThan(50);
  });

  it('TPOT is bandwidth-bound near W_bytes / (8 * BW_H100)', () => {
    const r = evaluate(llama8xH100({ workload: { batchSize: 1, inputLen: 2048, outputLen: 512 } }));
    // 141.2 GB / 8 / 3350 GB/s ~= 5.3 ms (KV adds a little)
    expect(r.tpotMs).toBeGreaterThan(4);
    expect(r.tpotMs).toBeLessThan(7);
  });

  it('E2E = TTFT + N_out * TPOT and throughput follows', () => {
    const r = evaluate(llama8xH100());
    expect(r.e2eMs).toBeCloseTo(r.ttftMs + 512 * r.tpotMs, 6);
    expect(r.throughputTps).toBeCloseTo((8 * 512) / (r.e2eMs / 1000), 6);
  });

  it('counts PP point-to-point comm in TTFT when exposed (multi-node)', () => {
    // 16 GPUs with gpusPerNode=8 => multi-node, so PP P2P uses inter-node BW.
    const spec: SystemSpec = {
      model: llama70b,
      gpu: h100,
      gpusPerNode: 8,
      interNodeBwGbps: 50,
      workload: { batchSize: 4, inputLen: 2048, outputLen: 64 },
      weightQuant: 'fp16',
      kvQuant: 'fp16',
      layout: { tp: 2, pp: 8, ep: 1, dp: 1 },
      flashAttention: true,
      headroom: 0.1,
    };
    const hidden = evaluate(spec, { ...IDEAL, ppCommOverlap: 1 });
    const exposed = evaluate(spec, { ...IDEAL, ppCommOverlap: 0 });
    // ppMsgBytes = B * N_in * h * b_act = 4*2048*8192*2; over 50 GB/s.
    const expectedPpCommMs = ((4 * 2048 * 8192 * 2) / (50 * 1e9)) * 1e3;
    expect(exposed.ttftMs - hidden.ttftMs).toBeCloseTo(expectedPpCommMs, 6);
  });

  it('MoE decode reads only covered experts (small batch)', () => {
    const spec: SystemSpec = {
      model: qwen3Moe,
      gpu: h100,
      gpusPerNode: 8,
      interNodeBwGbps: 50,
      workload: { batchSize: 1, inputLen: 1024, outputLen: 128 },
      weightQuant: 'fp16',
      kvQuant: 'fp16',
      layout: { tp: 2, pp: 1, ep: 4, dp: 1 },
      flashAttention: true,
      headroom: 0.1,
    };
    const r = evaluate(spec);
    expect(r.decode.expertCoverage).toBeCloseTo(8 / 128, 10);
    // non-expert 15.6 GB / tp2 + coverage * 454.4 GB / (tp2*ep4)
    const expectRead =
      (7.8e9 * 2) / 2 + ((235 - 7.8) * 1e9 * 2 * (8 / 128)) / 8;
    expect(r.decode.weightsReadBytes).toBeCloseTo(expectRead, -6);
  });

  it('ring all-reduce returns ms: bandwidth term + alpha', () => {
    const comm = {
      intraBwBps: 900 * 1e9,
      interBwBps: 50 * 1e9,
      alphaIntraMs: 0.006,
      alphaInterMs: 0.025,
    };
    // 1 MB message, tp=8: 2*(7/8)*1e6/(900e9) s = 1.944e-6 s = 0.001944 ms
    const t = ringAllreduceMs(1e6, 8, comm);
    expect(t).toBeCloseTo(0.001944 + 0.006, 6);
    expect(ringAllreduceMs(1e6, 1, comm)).toBe(0);
  });

  it('calibrated alpha lengthens decode TPOT but not ideal mode', () => {
    const base = llama8xH100({ workload: { batchSize: 1, inputLen: 2048, outputLen: 512 } });
    const ideal = evaluate(base);
    const withAlpha = evaluate(base, { ...IDEAL, tpCommOverlap: 0, alphaIntraMs: 0.006 });
    // decode TP message = B * h * b_act; bandwidth term is tiny but nonzero.
    const msg = 1 * 8192 * 2;
    const bwTermMs = ((2 * (8 - 1)) / 8) * (msg / (900 * 1e9)) * 1e3;
    const expectedExtraMs = 80 * 2 * (bwTermMs + 0.006);
    expect(withAlpha.tpotMs - ideal.tpotMs).toBeCloseTo(expectedExtraMs, 6);
  });

  it('PD disaggregation with ideal overlap adds zero KV transfer time', () => {
    const base = llama8xH100();
    const disagg = evaluate({
      ...base,
      disagg: {
        prefillGpus: 8,
        decodeGpus: 8,
        prefillLayout: { tp: 8, pp: 1, ep: 1, dp: 1 },
        decodeLayout: { tp: 8, pp: 1, ep: 1, dp: 1 },
        kvTransferOverlap: 1,
      },
    });
    const colocated = evaluate(base);
    expect(disagg.kvTransferExposedMs).toBe(0);
    expect(disagg.ttftMs).toBeCloseTo(colocated.ttftMs, 6);
    expect(disagg.tpotMs).toBeCloseTo(colocated.tpotMs, 6);
  });
});
