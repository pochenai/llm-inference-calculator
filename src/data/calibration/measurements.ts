// Structured measured-reference dataset (NVIDIA series).
//
// Each entry carries: protocol, source_url, scenario (model/gpu/layout/workload)
// and the measured metrics actually reported by the source. Missing metrics are
// omitted here and rendered as "-" by the comparison routine (src/data/calibrate.ts).
//
// protocol:
//   LAT - latency protocol, low-concurrency / single-request; directly comparable
//         with this calculator's single-batch model.
//   THR - throughput protocol, continuous-batching saturated steady state; only an
//         upper bound, compared against the decode roofline ceiling.

import type { ParallelLayout, QuantPrecision } from '../../core/types';

export interface MeasuredMetrics {
  ttftMs?: number;
  tpotMs?: number;
  throughputTps?: number;
  e2eMs?: number;
}

export interface Measurement {
  id: string;
  protocol: 'LAT' | 'THR';
  source: string;
  sourceUrl: string;
  modelId: string;
  gpuId: string;
  gpuCount: number;
  gpusPerNode: number;
  layout: ParallelLayout;
  weightQuant: QuantPrecision;
  kvQuant: QuantPrecision;
  batch: number;
  inputLen: number;
  outputLen: number;
  measured: MeasuredMetrics;
  note?: string;
}

export const MEASUREMENTS: Measurement[] = [
  // ---- Splitwise (Microsoft, NSDI'24): Llama-2-70B, TP=8, batch=1 [LAT] ----
  {
    id: 'splitwise-70b-h100-chat',
    protocol: 'LAT',
    source: 'Splitwise (arXiv 2311.18677)',
    sourceUrl: 'https://arxiv.org/html/2311.18677v2',
    modelId: 'llama2_70b',
    gpuId: 'h100_sxm',
    gpuCount: 8,
    gpusPerNode: 8,
    layout: { tp: 8, pp: 1, ep: 1, dp: 1 },
    weightQuant: 'fp16',
    kvQuant: 'fp16',
    batch: 1,
    inputLen: 1020,
    outputLen: 129,
    measured: { ttftMs: 84, tpotMs: 28, e2eMs: 3387 },
  },
  {
    id: 'splitwise-70b-h100-code',
    protocol: 'LAT',
    source: 'Splitwise (arXiv 2311.18677)',
    sourceUrl: 'https://arxiv.org/html/2311.18677v2',
    modelId: 'llama2_70b',
    gpuId: 'h100_sxm',
    gpuCount: 8,
    gpusPerNode: 8,
    layout: { tp: 8, pp: 1, ep: 1, dp: 1 },
    weightQuant: 'fp16',
    kvQuant: 'fp16',
    batch: 1,
    inputLen: 1500,
    outputLen: 13,
    measured: { ttftMs: 95, tpotMs: 31, e2eMs: 493 },
  },
  {
    id: 'splitwise-70b-a100-chat',
    protocol: 'LAT',
    source: 'Splitwise (arXiv 2311.18677)',
    sourceUrl: 'https://arxiv.org/html/2311.18677v2',
    modelId: 'llama2_70b',
    gpuId: 'a100_sxm_80g',
    gpuCount: 8,
    gpusPerNode: 8,
    layout: { tp: 8, pp: 1, ep: 1, dp: 1 },
    weightQuant: 'fp16',
    kvQuant: 'fp16',
    batch: 1,
    inputLen: 1020,
    outputLen: 129,
    measured: { ttftMs: 155, tpotMs: 40, e2eMs: 4957 },
  },
  {
    id: 'splitwise-70b-a100-code',
    protocol: 'LAT',
    source: 'Splitwise (arXiv 2311.18677)',
    sourceUrl: 'https://arxiv.org/html/2311.18677v2',
    modelId: 'llama2_70b',
    gpuId: 'a100_sxm_80g',
    gpuCount: 8,
    gpusPerNode: 8,
    layout: { tp: 8, pp: 1, ep: 1, dp: 1 },
    weightQuant: 'fp16',
    kvQuant: 'fp16',
    batch: 1,
    inputLen: 1500,
    outputLen: 13,
    measured: { ttftMs: 185, tpotMs: 52, e2eMs: 856 },
  },

  // ---- NVIDIA official blog: GPT-J 6B, H100 vs A100, TP=1 [LAT] ----
  {
    id: 'gptj-h100-b1',
    protocol: 'LAT',
    source: 'TensorRT-LLM H100 vs A100 blog',
    sourceUrl: 'https://nvidia.github.io/TensorRT-LLM/blogs/H100vsA100.html',
    modelId: 'gptj_6b',
    gpuId: 'h100_sxm',
    gpuCount: 1,
    gpusPerNode: 8,
    layout: { tp: 1, pp: 1, ep: 1, dp: 1 },
    weightQuant: 'fp8',
    kvQuant: 'fp8',
    batch: 1,
    inputLen: 128,
    outputLen: 128,
    measured: { tpotMs: 7.1, throughputTps: 185 },
  },
  {
    id: 'gptj-h100-b64',
    protocol: 'LAT',
    source: 'TensorRT-LLM H100 vs A100 blog',
    sourceUrl: 'https://nvidia.github.io/TensorRT-LLM/blogs/H100vsA100.html',
    modelId: 'gptj_6b',
    gpuId: 'h100_sxm',
    gpuCount: 1,
    gpusPerNode: 8,
    layout: { tp: 1, pp: 1, ep: 1, dp: 1 },
    weightQuant: 'fp8',
    kvQuant: 'fp8',
    batch: 64,
    inputLen: 128,
    outputLen: 128,
    measured: { ttftMs: 102, throughputTps: 10907 },
  },
  {
    id: 'gptj-a100-b1',
    protocol: 'LAT',
    source: 'TensorRT-LLM H100 vs A100 blog',
    sourceUrl: 'https://nvidia.github.io/TensorRT-LLM/blogs/H100vsA100.html',
    modelId: 'gptj_6b',
    gpuId: 'a100_sxm_80g',
    gpuCount: 1,
    gpusPerNode: 8,
    layout: { tp: 1, pp: 1, ep: 1, dp: 1 },
    weightQuant: 'fp16',
    kvQuant: 'fp16',
    batch: 1,
    inputLen: 128,
    outputLen: 128,
    measured: { tpotMs: 12.5, throughputTps: 111 },
  },
  {
    id: 'gptj-a100-b64',
    protocol: 'LAT',
    source: 'TensorRT-LLM H100 vs A100 blog',
    sourceUrl: 'https://nvidia.github.io/TensorRT-LLM/blogs/H100vsA100.html',
    modelId: 'gptj_6b',
    gpuId: 'a100_sxm_80g',
    gpuCount: 1,
    gpusPerNode: 8,
    layout: { tp: 1, pp: 1, ep: 1, dp: 1 },
    weightQuant: 'fp16',
    kvQuant: 'fp16',
    batch: 64,
    inputLen: 128,
    outputLen: 128,
    measured: { ttftMs: 481, throughputTps: 3679 },
  },

  // ---- llama.cpp single GPU, batch=1 [LAT]; Q4_K_M approximated as int4 ----
  {
    id: 'llamacpp-h100-8b-f16',
    protocol: 'LAT',
    source: 'XiongjieDai GPU-Benchmarks-on-LLM-Inference',
    sourceUrl: 'https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference',
    modelId: 'llama3_1_8b',
    gpuId: 'h100_pcie',
    gpuCount: 1,
    gpusPerNode: 8,
    layout: { tp: 1, pp: 1, ep: 1, dp: 1 },
    weightQuant: 'fp16',
    kvQuant: 'fp16',
    batch: 1,
    inputLen: 512,
    outputLen: 512,
    measured: { ttftMs: 47.3, tpotMs: 14.7 },
    note: 'ttft derived from pp=10816 tok/s; tpot from tg=68 tok/s',
  },
  {
    id: 'llamacpp-h100-70b-q4',
    protocol: 'LAT',
    source: 'XiongjieDai GPU-Benchmarks-on-LLM-Inference',
    sourceUrl: 'https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference',
    modelId: 'llama3_3_70b',
    gpuId: 'h100_pcie',
    gpuCount: 1,
    gpusPerNode: 8,
    layout: { tp: 1, pp: 1, ep: 1, dp: 1 },
    weightQuant: 'int4',
    kvQuant: 'fp16',
    batch: 1,
    inputLen: 512,
    outputLen: 512,
    measured: { ttftMs: 505, tpotMs: 40 },
    note: 'Q4_K_M (~0.60 B/param) approximated as int4 (0.5); ttft from pp=1013, tpot from tg=25',
  },
  {
    id: 'llamacpp-a100-70b-q4',
    protocol: 'LAT',
    source: 'XiongjieDai GPU-Benchmarks-on-LLM-Inference',
    sourceUrl: 'https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference',
    modelId: 'llama3_3_70b',
    gpuId: 'a100_sxm_80g',
    gpuCount: 1,
    gpusPerNode: 8,
    layout: { tp: 1, pp: 1, ep: 1, dp: 1 },
    weightQuant: 'int4',
    kvQuant: 'fp16',
    batch: 1,
    inputLen: 512,
    outputLen: 512,
    measured: { ttftMs: 626, tpotMs: 40.7 },
    note: 'ttft from pp=818, tpot from tg=24.6',
  },

  // ---- LMSYS chunked pipeline: Qwen3-235B-FP8 on H20, batch=1, 128K prefill [LAT, prefill-only] ----
  {
    id: 'lmsys-qwen3-235b-pp1tp4',
    protocol: 'LAT',
    source: 'LMSYS chunked-pipeline blog 2026-01-15',
    sourceUrl: 'https://www.lmsys.org/blog/2026-01-15-chunked-pipeline/',
    modelId: 'qwen3_235b',
    gpuId: 'h20',
    gpuCount: 4,
    gpusPerNode: 8,
    layout: { tp: 4, pp: 1, ep: 1, dp: 1 },
    weightQuant: 'fp8',
    kvQuant: 'fp8',
    batch: 1,
    inputLen: 131072,
    outputLen: 1,
    measured: { ttftMs: 56000, throughputTps: 2350 },
    note: 'prefill-only; throughput = prefill input tok/s (static chunk 6144)',
  },
  {
    id: 'lmsys-qwen3-235b-pp4tp4',
    protocol: 'LAT',
    source: 'LMSYS chunked-pipeline blog 2026-01-15',
    sourceUrl: 'https://www.lmsys.org/blog/2026-01-15-chunked-pipeline/',
    modelId: 'qwen3_235b',
    gpuId: 'h20',
    gpuCount: 16,
    gpusPerNode: 8,
    layout: { tp: 4, pp: 4, ep: 1, dp: 1 },
    weightQuant: 'fp8',
    kvQuant: 'fp8',
    batch: 1,
    inputLen: 131072,
    outputLen: 1,
    measured: { ttftMs: 17700, throughputTps: 8100 },
    note: 'prefill-only; DCK adaptive chunk',
  },
  {
    id: 'lmsys-qwen3-235b-pp8tp4',
    protocol: 'LAT',
    source: 'LMSYS chunked-pipeline blog 2026-01-15',
    sourceUrl: 'https://www.lmsys.org/blog/2026-01-15-chunked-pipeline/',
    modelId: 'qwen3_235b',
    gpuId: 'h20',
    gpuCount: 32,
    gpusPerNode: 8,
    layout: { tp: 4, pp: 8, ep: 1, dp: 1 },
    weightQuant: 'fp8',
    kvQuant: 'fp8',
    batch: 1,
    inputLen: 131072,
    outputLen: 1,
    measured: { ttftMs: 10500, throughputTps: 14600 },
    note: 'prefill-only; DCK adaptive chunk',
  },

  // ---- NVIDIA TRT-LLM official overview: saturated steady state [THR] ----
  {
    id: 'trtllm-8b-h100-1024-2048',
    protocol: 'THR',
    source: 'TensorRT-LLM Performance Overview',
    sourceUrl: 'https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html',
    modelId: 'llama3_1_8b',
    gpuId: 'h100_sxm',
    gpuCount: 1,
    gpusPerNode: 8,
    layout: { tp: 1, pp: 1, ep: 1, dp: 1 },
    weightQuant: 'fp8',
    kvQuant: 'fp8',
    batch: 0, // batch swept to saturation by the source
    inputLen: 1024,
    outputLen: 2048,
    measured: { throughputTps: 13166 },
  },
  {
    id: 'trtllm-70b-h100-1024-2048',
    protocol: 'THR',
    source: 'TensorRT-LLM Performance Overview',
    sourceUrl: 'https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html',
    modelId: 'llama3_3_70b',
    gpuId: 'h100_sxm',
    gpuCount: 2,
    gpusPerNode: 8,
    layout: { tp: 2, pp: 1, ep: 1, dp: 1 },
    weightQuant: 'fp8',
    kvQuant: 'fp8',
    batch: 0,
    inputLen: 1024,
    outputLen: 2048,
    measured: { throughputTps: 3785 },
  },
  {
    id: 'trtllm-405b-h100-1024-2048',
    protocol: 'THR',
    source: 'TensorRT-LLM Performance Overview',
    sourceUrl: 'https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html',
    modelId: 'llama3_405b',
    gpuId: 'h100_sxm',
    gpuCount: 8,
    gpusPerNode: 8,
    layout: { tp: 8, pp: 1, ep: 1, dp: 1 },
    weightQuant: 'fp8',
    kvQuant: 'fp8',
    batch: 0,
    inputLen: 1024,
    outputLen: 2048,
    measured: { throughputTps: 3237 },
  },
];
