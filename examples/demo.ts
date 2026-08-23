// Smoke demo: evaluate a few representative scenarios in ideal-value mode.
// Models / GPUs come from the bundled data catalog (src/data).

import { evaluate } from '../src/core/metrics';
import type { SystemSpec } from '../src/core/types';
import { model } from '../src/data/models';
import { gpu } from '../src/data/gpus/nvidia';

const llama70b = model('llama3_1_70b');
const qwen3Moe = model('qwen3_235b');
const h100 = gpu('h100_sxm');

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
