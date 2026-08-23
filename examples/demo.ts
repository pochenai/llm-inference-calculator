// Smoke demo: evaluate a few representative scenarios in ideal-value mode.

import type { GpuSpec, ModelSpec, SystemSpec } from '../src/core/types';
import { evaluate } from '../src/core/metrics';

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

const h100: GpuSpec = {
  id: 'h100_sxm',
  name: 'H100 SXM5',
  vramGb: 80,
  bwGbps: 3350,
  peakTflops: { bf16: 989, fp8: 1979, int8: 1979, int4: 3958 },
  nvlinkBwGbps: 900,
};

function report(title: string, spec: SystemSpec): void {
  const outcome = evaluate(spec);
  console.log(`\n=== ${title} ===`);
  if (!outcome.ok) {
    console.log(`ERROR [${outcome.error.code}]: ${outcome.error.message}`);
    return;
  }
  const r = outcome.value;
  const gb = (x: number): string => (x / 1e9).toFixed(1) + ' GB';
  console.log(`feasible=${r.feasible}  weights/gpu=${gb(r.memory.weightsBytes)}  kv=${gb(r.memory.kvBytes)}  B_max=${r.memory.bMax ?? 'n/a'}`);
  console.log(`TTFT=${r.ttftMs.toFixed(1)} ms  TPOT=${r.tpotMs.toFixed(2)} ms  E2E=${(r.e2eMs / 1000).toFixed(2)} s  throughput=${r.throughputTps.toFixed(0)} tok/s`);
}

report('Llama 70B fp16, 8xH100 TP=8, B=8, in=2048/out=512', {
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
});

report('Llama 70B fp8 weights, 8xH100 TP=8, B=32, in=4096/out=1024', {
  model: llama70b,
  gpu: h100,
  gpusPerNode: 8,
  interNodeBwGbps: 50,
  workload: { batchSize: 32, inputLen: 4096, outputLen: 1024 },
  weightQuant: 'fp8',
  kvQuant: 'fp8',
  layout: { tp: 8, pp: 1, ep: 1, dp: 1 },
  flashAttention: true,
  headroom: 0.1,
});

report('Qwen3 235B MoE fp16, 16xH100 (2 nodes) TP=2 EP=4 PP=1 DP=2, B=16, in=2048/out=512', {
  model: qwen3Moe,
  gpu: h100,
  gpusPerNode: 8,
  interNodeBwGbps: 50,
  workload: { batchSize: 16, inputLen: 2048, outputLen: 512 },
  weightQuant: 'fp16',
  kvQuant: 'fp16',
  layout: { tp: 2, pp: 1, ep: 4, dp: 2 },
  flashAttention: true,
  headroom: 0.1,
});

report('Qwen3 235B MoE fp8, 8xH100 TP=2 EP=4, B=8, in=2048/out=512', {
  model: qwen3Moe,
  gpu: h100,
  gpusPerNode: 8,
  interNodeBwGbps: 50,
  workload: { batchSize: 8, inputLen: 2048, outputLen: 512 },
  weightQuant: 'fp8',
  kvQuant: 'fp8',
  layout: { tp: 2, pp: 1, ep: 4, dp: 1 },
  flashAttention: true,
  headroom: 0.1,
});
