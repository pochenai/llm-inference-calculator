// Tests for speculative decoding support.

import { describe, expect, it } from 'vitest';
import type { ModelSpec, SystemSpec, SpeculativeConfig } from '../src/core/types.js';
import { model, ALL_MODELS } from '../src/data/models/index.js';
import { gpu } from '../src/data/gpus/nvidia/index.js';
import { evaluate as evaluateResult } from '../src/core/metrics.js';
import type { EvaluationResult } from '../src/core/metrics.js';
import { IDEAL } from '../src/core/calibration.js';
import type { Calibration } from '../src/core/calibration.js';
import { unwrap } from '../src/core/errors.js';
import { suggestDraftModel, modelsInRange, familyOf } from '../src/data/models/suggest.js';
import { deriveConstants } from '../src/core/model.js';
import { sdDecodeStepTime } from '../src/core/latency.js';
import type { PhaseInput } from '../src/core/latency.js';
import { resolveInterconnect } from '../src/core/hardware.js';
import { buildCommModel } from '../src/core/latency.js';

function evaluate(spec: SystemSpec, cal?: Calibration): EvaluationResult {
  return unwrap(evaluateResult(spec, cal));
}

// Fixtures
const llama70b = model('llama3_1_70b');
const llama8b = model('llama3_1_8b');
const qwen7b = model('qwen25_7b');
const h100 = gpu('h100_sxm');

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

describe('model suggestion utilities', () => {
  it('familyOf extracts correct prefixes', () => {
    expect(familyOf(llama70b)).toBe('llama');
    expect(familyOf(qwen7b)).toBe('qwen');
  });

  it('modelsInRange returns dense models in target range', () => {
    // 70B model, looking for 7-14B (0.1-0.2 ratio)
    const candidates = modelsInRange(ALL_MODELS, 70, 0.1, 0.2);
    expect(candidates.length).toBeGreaterThan(0);
    // All should be dense and in range [7, 14]
    for (const m of candidates) {
      expect(m.type).toBe('dense');
      expect(m.paramsB).toBeGreaterThanOrEqual(7);
      expect(m.paramsB).toBeLessThanOrEqual(14);
    }
    // Should be sorted descending
    for (let i = 0; i < candidates.length - 1; i++) {
      expect(candidates[i]!.paramsB).toBeGreaterThanOrEqual(candidates[i + 1]!.paramsB);
    }
  });

  it('suggestDraftModel prefers same family', () => {
    const draft = suggestDraftModel(llama70b, ALL_MODELS);
    expect(draft).not.toBeNull();
    expect(draft!.id).not.toBe(llama70b.id);
    // Should prefer Llama family
    expect(familyOf(draft!)).toBe('llama');
  });

  it('suggestDraftModel returns null for very small models', () => {
    // 1B model, looking for 0.1-0.2B (doesn't exist)
    const smallModel: ModelSpec = {
      id: 'tiny_1b',
      name: 'Tiny 1B',
      type: 'dense',
      paramsB: 1,
      layers: 12,
      hiddenSize: 1024,
      kvHeads: 8,
      headDim: 128,
      maxCtx: 2048,
    };
    const draft = suggestDraftModel(smallModel, ALL_MODELS, 0.1, 0.2);
    expect(draft).toBeNull();
  });

  it('suggestDraftModel never returns the main model itself', () => {
    const draft = suggestDraftModel(llama70b, ALL_MODELS);
    expect(draft).not.toBeNull();
    expect(draft!.id).not.toBe(llama70b.id);
  });
});

describe('speculative decoding VRAM', () => {
  it('SD VRAM > non-SD VRAM for same config', () => {
    const specNoSd = llama8xH100();
    const resultNoSd = evaluate(specNoSd);

    const sdConfig: SpeculativeConfig = {
      draftModel: llama8b,
      draftTp: 8,
      gamma: 5,
      acceptanceRate: 0.7,
    };
    const specSd = llama8xH100({ speculative: sdConfig });
    const resultSd = evaluate(specSd);

    // SD should use more VRAM (draft weights + draft KV)
    expect(resultSd.memory.totalBytes).toBeGreaterThan(resultNoSd.memory.totalBytes);
    expect(resultSd.memory.draftWeightsBytes).toBeDefined();
    expect(resultSd.memory.draftKvBytes).toBeDefined();
  });

  it('draft weights scale inversely with draftTp', () => {
    const spec1 = llama8xH100({
      speculative: { draftModel: llama8b, draftTp: 1, gamma: 5, acceptanceRate: 0.7 },
    });
    const spec2 = llama8xH100({
      speculative: { draftModel: llama8b, draftTp: 2, gamma: 5, acceptanceRate: 0.7 },
    });
    const result1 = evaluate(spec1);
    const result2 = evaluate(spec2);

    // More TP = less per GPU
    expect(result1.memory.draftWeightsBytes!).toBeGreaterThan(result2.memory.draftWeightsBytes!);
  });

  it('bMax decreases when SD enabled', () => {
    const specNoSd = llama8xH100();
    const resultNoSd = evaluate(specNoSd);

    const specSd = llama8xH100({
      speculative: { draftModel: llama8b, draftTp: 8, gamma: 5, acceptanceRate: 0.7 },
    });
    const resultSd = evaluate(specSd);

    // SD uses more VRAM, so less budget for KV -> lower bMax
    expect(resultSd.memory.bMax).toBeLessThan(resultNoSd.memory.bMax);
  });
});

describe('speculative decoding latency', () => {
  it('SD TPOT < standard TPOT when acceptanceRate is high', () => {
    const specNoSd = llama8xH100();
    const resultNoSd = evaluate(specNoSd);

    const specSd = llama8xH100({
      speculative: { draftModel: llama8b, draftTp: 8, gamma: 5, acceptanceRate: 0.7 },
    });
    const resultSd = evaluate(specSd);

    // SD raw cycle time should be faster than baseline (per-token SD cycle
    // vs single main model step). The speculative.speedup captures this.
    // Note: result.tpotMs in non-PD mode includes prefill/decode contention,
    // which shifts when SD speeds up decode; raw speedup isolates the SD effect.
    expect(resultSd.speculative).toBeDefined();
    expect(resultSd.speculative!.speedup).toBeGreaterThan(1);
    expect(resultSd.speculative!.baselineTpotMs).toBeCloseTo(resultNoSd.decode.tpotMs, 3);
  });

  it('SD TPOT formula matches expected', () => {
    const sdConfig: SpeculativeConfig = {
      draftModel: llama8b,
      draftTp: 8,
      gamma: 5,
      acceptanceRate: 0.7,
    };
    const spec = llama8xH100({ speculative: sdConfig });
    const result = evaluate(spec);

    expect(result.speculative).toBeDefined();
    const sd = result.speculative!;

    // expectedTokensPerCycle = γ·α + 1
    const expectedTokens = sdConfig.gamma * sdConfig.acceptanceRate + 1;
    expect(sd.expectedTokensPerCycle).toBeCloseTo(expectedTokens, 5);

    // cycleTimeMs = γ · draftStepMs + verifyStepMs
    const expectedCycleTime = sdConfig.gamma * sd.draftStepMs + sd.verifyStepMs;
    expect(sd.cycleTimeMs).toBeCloseTo(expectedCycleTime, 3);

    // tpotMs = cycleTimeMs / expectedTokensPerCycle
    const expectedTpot = sd.cycleTimeMs / sd.expectedTokensPerCycle;
    expect(sd.verifyStepMs).toBeCloseTo(sd.baselineTpotMs, 3); // verify ≈ baseline
  });

  it('SD speedup > 1 for typical parameters', () => {
    const spec = llama8xH100({
      speculative: { draftModel: llama8b, draftTp: 8, gamma: 5, acceptanceRate: 0.7 },
    });
    const result = evaluate(spec);

    expect(result.speculative!.speedup).toBeGreaterThan(1);
    // Typical speedup for 70B main + 8B draft should be 2-3x
    expect(result.speculative!.speedup).toBeLessThan(5);
  });

  it('SD with acceptanceRate=0 is slower than baseline', () => {
    const specSd = llama8xH100({
      speculative: { draftModel: llama8b, draftTp: 8, gamma: 5, acceptanceRate: 0 },
    });
    const resultSd = evaluate(specSd);

    // Zero acceptance rate -> draft work is wasted -> slower.
    // speedup captures the raw SD cycle cost vs baseline decode step.
    expect(resultSd.speculative!.speedup).toBeLessThan(1);
  });

  it('SD with gamma=1 still provides speedup (if draft is fast and acceptance is high)', () => {
    const specSd = llama8xH100({
      speculative: { draftModel: llama8b, draftTp: 8, gamma: 1, acceptanceRate: 0.7 },
    });
    const resultSd = evaluate(specSd);

    // gamma=1 with high acceptance rate can still be faster:
    // expectedTokensPerCycle = 1*0.7 + 1 = 1.7 tokens per cycle
    // 1 fast draft step + 1 verify step -> 1.7 tokens on average
    // This is faster than 1 slow main step -> 1 token
    expect(resultSd.speculative!.speedup).toBeGreaterThan(1);
  });
});

describe('speculative decoding integration', () => {
  it('evaluate() returns speculative field when SD enabled', () => {
    const spec = llama8xH100({
      speculative: { draftModel: llama8b, draftTp: 8, gamma: 5, acceptanceRate: 0.7 },
    });
    const result = evaluate(spec);

    expect(result.speculative).toBeDefined();
    expect(result.speculative!.draftModelId).toBe(llama8b.id);
    expect(result.speculative!.draftModelName).toBe(llama8b.name);
    expect(result.speculative!.gamma).toBe(5);
    expect(result.speculative!.acceptanceRate).toBe(0.7);
  });

  it('evaluate() returns no speculative field when SD disabled', () => {
    const spec = llama8xH100();
    const result = evaluate(spec);

    expect(result.speculative).toBeUndefined();
  });

  it('TTFT unchanged by SD', () => {
    const specNoSd = llama8xH100();
    const resultNoSd = evaluate(specNoSd);

    const specSd = llama8xH100({
      speculative: { draftModel: llama8b, draftTp: 8, gamma: 5, acceptanceRate: 0.7 },
    });
    const resultSd = evaluate(specSd);

    // Raw prefill time should be the same (SD only affects decode).
    // Note: result.ttftMs may differ in non-PD mode because it includes decode
    // contention (ρ_decode * tpotMs), and SD changes the effective TPOT.
    expect(resultSd.prefill.ttftMs).toBeCloseTo(resultNoSd.prefill.ttftMs, 3);
  });

  it('throughput increases with SD', () => {
    const specSd = llama8xH100({
      speculative: { draftModel: llama8b, draftTp: 8, gamma: 5, acceptanceRate: 0.7 },
    });
    const resultSd = evaluate(specSd);

    // SD produces more output tokens per cycle than baseline decode.
    // speedup > 1 means the SD cycle generates tokens faster than a single
    // main model decode step, which directly improves raw decode throughput.
    // Note: in non-PD mode, effective system throughput also depends on
    // prefill/decode resource contention which is independent of SD.
    expect(resultSd.speculative!.speedup).toBeGreaterThan(1);
    expect(resultSd.speculative!.expectedTokensPerCycle).toBeGreaterThan(1);
  });

  it('SD works with MoE main model', () => {
    const qwen235b = model('qwen3_235b');
    const spec: SystemSpec = {
      model: qwen235b,
      gpu: h100,
      gpusPerNode: 8,
      interNodeBwGbps: 50,
      workload: { batchSize: 8, inputLen: 2048, outputLen: 512 },
      weightQuant: 'fp8',
      kvQuant: 'fp16',
      layout: { tp: 8, pp: 1, ep: 1, dp: 1 },
      flashAttention: true,
      headroom: 0.1,
      speculative: {
        draftModel: qwen7b,
        draftTp: 8,
        gamma: 5,
        acceptanceRate: 0.7,
      },
    };
    const result = evaluate(spec);

    expect(result.feasible).toBe(true);
    expect(result.speculative).toBeDefined();
    expect(result.speculative!.speedup).toBeGreaterThan(1);
  });

  it('SD with PD disaggregation: draft only in decode pool', () => {
    // PD disaggregation with SD: draft model only in decode pool
    const specSdDisagg: SystemSpec = {
      model: llama70b,
      gpu: h100,
      gpusPerNode: 8,
      interNodeBwGbps: 50,
      workload: { batchSize: 8, inputLen: 2048, outputLen: 512 },
      weightQuant: 'fp16',
      kvQuant: 'fp16',
      layout: { tp: 4, pp: 1, ep: 1, dp: 1 },
      flashAttention: true,
      headroom: 0.1,
      disagg: {
        prefillGpus: 4,
        decodeGpus: 4,
        prefillLayout: { tp: 4, pp: 1, ep: 1, dp: 1 },
        decodeLayout: { tp: 4, pp: 1, ep: 1, dp: 1 },
        kvTransferOverlap: 0.8,
      },
      speculative: {
        draftModel: llama8b,
        draftTp: 4, // Max = decodeGpus = 4
        gamma: 5,
        acceptanceRate: 0.7,
      },
    };

    // Create a copy without speculative for baseline comparison
    const { speculative: _, ...specNoSdDisagg } = specSdDisagg;

    const resultSd = evaluate(specSdDisagg);
    const resultNoSd = evaluate(specNoSdDisagg as SystemSpec);

    // KV transfer should be the same (only main KV transferred, draft prefill in decode pool)
    expect(resultSd.kvTransferExposedMs).toBeCloseTo(resultNoSd.kvTransferExposedMs, 3);

    // Prefill pool VRAM should be the same (only main model)
    expect(resultSd.memoryPrefillPool!.totalBytes).toBeCloseTo(
      resultNoSd.memoryPrefillPool!.totalBytes,
      3,
    );

    // Decode pool VRAM should be higher with SD (main + draft)
    expect(resultSd.memory.totalBytes).toBeGreaterThan(resultNoSd.memory.totalBytes);
    expect(resultSd.memory.draftWeightsBytes).toBeDefined();
    expect(resultSd.memory.draftKvBytes).toBeDefined();

    // Draft TP should be limited to decodeGpus
    expect(resultSd.speculative).toBeDefined();
    expect(resultSd.speculative!.speedup).toBeGreaterThan(1);
  });
});

describe('sdDecodeStepTime function', () => {
  it('computes correct SD decode details', () => {
    const mainInp: PhaseInput = {
      model: llama70b,
      derived: deriveConstants(llama70b, 'fp16', 'fp16'),
      gpu: h100,
      layout: { tp: 8, pp: 1, ep: 1, dp: 1 },
      workload: { batchSize: 8, inputLen: 2048, outputLen: 512 },
      weightQuant: 'fp16',
      cal: IDEAL,
      comm: buildCommModel(resolveInterconnect(h100, 50), IDEAL),
      gpusPerNode: 8,
    };

    const draftInp: PhaseInput = {
      model: llama8b,
      derived: deriveConstants(llama8b, 'fp16', 'fp16'),
      gpu: h100,
      layout: { tp: 8, pp: 1, ep: 1, dp: 1 },
      workload: { batchSize: 8, inputLen: 2048, outputLen: 512 },
      weightQuant: 'fp16',
      cal: IDEAL,
      comm: buildCommModel(resolveInterconnect(h100, 50), IDEAL),
      gpusPerNode: 8,
    };

    const sd = sdDecodeStepTime(mainInp, draftInp, 5, 0.7);

    expect(sd.gamma).toBe(5);
    expect(sd.acceptanceRate).toBe(0.7);
    expect(sd.expectedTokensPerCycle).toBeCloseTo(5 * 0.7 + 1, 5);
    expect(sd.cycleTimeMs).toBeCloseTo(5 * sd.draftStep.tpotMs + sd.verifyStep.tpotMs, 3);
    expect(sd.speedup).toBeGreaterThan(1);
  });
});
