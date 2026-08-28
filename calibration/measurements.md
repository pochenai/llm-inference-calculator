# Benchmark Data Archive (NVIDIA Series)

Collected 2026-08-23. This file is the **human-readable archive** of publicly available LLM inference benchmarks, maintained for provenance and interpretation context. The structured, machine-readable version lives in [`../src/data/measurements.ts`](../src/data/measurements.ts).

For how these numbers are compared against the calculator's ideal values, and how to derive calibration factors, see [`README.md`](./README.md).

---

## 1. How to Read This File

**Notation**: ISL / OSL = Input / Output Sequence Length (token counts), equivalent to this project's `workload.inputLen` / `workload.outputLen`.

**Protocol tags** (each source is classified):

| Tag | Meaning | Directly comparable? |
|---|---|---|
| `LAT` | Latency protocol — low-concurrency or single-request | ✅ Yes, matches single-batch model |
| `THR` | Throughput protocol — continuous-batching saturated steady state | ⚠️ Upper-bound reference only |
| `CFG` | Deployment config only, no performance numbers | ❌ No comparison |

---

## 2. NVIDIA Official Benchmarks

NVIDIA publishes two types of data: TRT-LLM performance overview tables (saturated throughput) and blog posts with direct hardware comparisons. Both serve as the primary hardware-level reference.

### 2.1 TRT-LLM Performance Overview [THR]

Source: [TensorRT-LLM Performance Overview](https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html).
Metric: Total Output Throughput (tok/s), continuous batching with batch swept to saturation.

#### Llama 3.1 8B FP8, 1×GPU (TP=1)

| ISL/OSL | H100 | H200 |
|---|---|---|
| 128/128 | 26,401 | 27,028 |
| 128/2048 | 21,413 | 23,102 |
| 1024/2048 | 13,166 | 16,058 |
| 2048/128 | 3,276 | 3,391 |
| 2048/2048 | 9,462 | 11,822 |

#### Llama 3.3 70B FP8, 2×GPU (TP=2)

| ISL/OSL | H100 | H200 |
|---|---|---|
| 128/128 | 6,092 | 6,328 |
| 128/2048 | 5,893 | 7,467 |
| 1024/2048 | 3,785 | 5,480 |
| 2048/128 | 723 | 748 |
| 2048/2048 | 2,786 | 3,776 |

#### Llama 3.1 405B FP8, 8×GPU (TP=8)

| ISL/OSL | H100 | H200 |
|---|---|---|
| 128/128 | 3,705 | — |
| 128/2048 | 4,517 | 4,715 |
| 1024/2048 | 3,237 | 3,610 |
| 2048/128 | 433 | 441 |
| 2048/2048 | 2,217 | 2,841 |

#### Llama 4 Maverick (MoE 400B/17B active) FP8, 8×GPU (TP=8)

| ISL/OSL | H100 | H200 |
|---|---|---|
| 128/4096 | 11,163 | 18,541 |
| 1024/2048 | 11,584 | 16,859 |
| 2048/128 | 3,832 | 4,364 |

#### Llama 3.3 70B FP4, 1×B200 / 1×GB200 (TP=1)

| ISL/OSL | B200 | GB200 |
|---|---|---|
| 128/128 | 10,614 | 11,101 |
| 128/2048 | 9,446 | 10,276 |
| 1024/2048 | 6,547 | 7,923 |
| 2048/128 | 1,330 | 1,418 |
| 2048/2048 | 4,528 | 5,327 |

#### Llama 3.1 405B FP4, 4×B200 / 4×GB200 (TP=4)

| ISL/OSL | B200 | GB200 |
|---|---|---|
| 128/128 | 6,219 | 6,599 |
| 128/2048 | 7,178 | 7,497 |
| 1024/2048 | 4,833 | 4,686 |
| 2048/128 | 738 | 762 |
| 2048/2048 | 4,024 | 4,327 |

### 2.2 NVIDIA Blog: H100 vs A100 — GPT-J 6B [LAT + THR]

Source: [H100 has 4.6× A100 performance in TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/blogs/H100vsA100.html).
TRT-LLM v0.5.0; H100 FP8 / A100 FP16, SXM 80G, TP=1.

This is one of the few sources reporting **both latency and throughput** for the same hardware, making it valuable for cross-validating the prefill and decode models independently.

| Scenario | H100 | A100 |
|---|---|---|
| batch=1, ISL=128 decode | 185 tok/s (7.1 ms/token) | 111 tok/s (12.5 ms/token) |
| batch=64, ISL/OSL=128/128 | 10,907 tok/s, TTFT 102 ms | 3,679 tok/s, TTFT 481 ms |

### 2.3 MLPerf Inference v5.0 (NVIDIA Submission) [THR]

Source: [NVIDIA MLPerf v5.0 blog](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0/).
Llama 2 70B, offline (batch saturated):

| System | Throughput (tok/s) |
|---|---|
| 8×H200 | 34,988 |
| 8×B200 | 98,858 |
| GB200 NVL72 | 869,203 |

---

## 3. Academic Papers

Peer-reviewed work provides the most rigorous latency measurements with controlled variables.

### 3.1 Splitwise (Microsoft, NSDI'24): Llama-2-70B, TP=8 [LAT + PD]

Source: [arXiv 2311.18677](https://arxiv.org/html/2311.18677v2).
DGX node, 8 GPUs, TP=8, InfiniBand (A100 pair 200 Gbps / H100 pair 400 Gbps), single request P50.

This is the primary source for **low-concurrency latency** on 70B-class models across A100 and H100, with both prefill-heavy (coding) and decode-heavy (conversation) workloads.

| Workload | Hardware | TTFT | TBT (=TPOT) | E2E |
|---|---|---|---|---|
| coding (in≈1500 / out=13) | 8×A100 | 185 ms | 52 ms | 856 ms |
| coding (in≈1500 / out=13) | 8×H100 | 95 ms | 31 ms | 493 ms |
| conversation (in≈1020 / out=129) | 8×A100 | 155 ms | 40 ms | 4,957 ms |
| conversation (in≈1020 / out=129) | 8×H100 | 84 ms | 28 ms | 3,387 ms |

**PD-disaggregation findings** (unique to this paper):

- Async KV-cache transfer adds only **0.8%** to E2E; serialized handoff adds up to +3%
- Prefill-side efficiency degrades once combined input exceeds 2,048 tokens
- Decode-side batch saturates memory at 64
- Example optimal splits: coding → 35 prefill + 5 decode machines; chat → 25 + 15

---

## 4. Industry Benchmarks

Third-party measurements from production-oriented platforms. These reflect real-world serving conditions but may have less controlled variables than academic papers.

### 4.1 Koyeb: Llama 3.1 8B and Qwen 2.5 7B, Single GPU [LAT]

Source: [Koyeb GPU benchmarks](https://www.koyeb.com/docs/hardware/gpu-benchmarks).
Methodology: vLLM benchmarking CLI, synthetic random data, single GPU, no precision info (assumed bf16/fp16). TP=1.

This is the most systematic single-GPU dataset, sweeping across **3 GPUs × 3 batch sizes × 3 sequence shapes** — ideal for validating how the model captures batch and sequence scaling on a single device.

#### Llama 3.1 8B Instruct, 1×GPU (TP=1)

| ISL/OSL | Batch | H200 SXM | H100 SXM | A100 SXM 80G |
|---|---|---|---|---|
| 512/512 | 1 | 169 | 99 | 86 |
| 512/512 | 8 | 1,309 | 816 | 652 |
| 512/512 | 32 | 4,621 | 3,008 | 2,083 |
| 1024/1024 | 1 | 168 | 99 | 86 |
| 1024/1024 | 8 | 1,289 | 722 | 632 |
| 1024/1024 | 32 | 4,419 | 2,401 | 1,888 |
| 4096/1024 | 1 | 164 | 99 | 83 |
| 4096/1024 | 8 | 1,162 | 616 | 544 |
| 4096/1024 | 32 | 3,209 | 1,591 | 1,202 |

#### Qwen 2.5 7B Instruct, 1×GPU (TP=1)

| ISL/OSL | Batch | H200 SXM | H100 SXM | A100 SXM 80G |
|---|---|---|---|---|
| 512/512 | 1 | 182 | 105 | 93 |
| 512/512 | 8 | 1,371 | 808 | 699 |
| 512/512 | 32 | 4,523 | 2,937 | 2,486 |
| 1024/1024 | 1 | 182 | 106 | 92 |
| 1024/1024 | 8 | 1,368 | 800 | 682 |
| 1024/1024 | 32 | 4,719 | 2,802 | 2,363 |
| 4096/1024 | 1 | 180 | 104 | 90 |
| 4096/1024 | 8 | 1,143 | 750 | 636 |
| 4096/1024 | 32 | 1,253 | 2,156 | 1,913 |

Note: DeepSeek R1 data was skipped (identical to Llama 3.1 8B — likely a copy error in the source). RTX PRO 6000 GPU was skipped (not in the GPU catalog).

### 4.2 Spheron: Multi-GPU Cluster Throughput [THR]

Source: [Spheron — Best GPU for AI Inference 2026](https://www.spheron.network/blog/best-gpu-for-ai-inference-2026/).
Llama 3.3 70B, 8-GPU cluster, saturated steady state.

This source uniquely covers **H100 / H200 / B200 with both FP8 and FP4** on the same workload, enabling direct cross-GPU and cross-precision comparisons.

| GPU | Precision | tok/s (8× cluster) | $/hr | $/M tokens |
|---|---|---|---|---|
| H100 SXM | FP8 | ~24,528 | $1.03 | ~$0.095 |
| H200 SXM | FP8 | ~34,992 | $4.54 | ~$0.288 |
| B200 | FP8 | ~55,776 | $6.02 | ~$0.239 |
| B200 | FP4 | ~102,728 | $2.12 | — |

Note: Source reports per-GPU tok/s; values above are cluster totals (× 8). Benchmark appears prefill-dominated (prefillRatio ≈ 1), which explains the very high throughput ratios when compared against the single-batch model.

### 4.3 Other Third-Party Sources [THR]

- [dlewis.io: Llama 3.3 70B, 4×H100, NIM/TRT-LLM, bf16](https://dlewis.io/evaluating-llama-33-70b-inference-h100-a100/):
  200→200 peak ~7,000 TPS; 250 concurrent 1000→200 ~2,600 TPS; under load TTFT < 5 s (queueing-dominated).

- [Cerebrium: Llama 3.1 70B FP8](https://cerebrium.ai/blog/benchmarking-vllm-sglang-tensorrt-for-llama-3-1-api):
  batch=1, in=256 TTFT: vLLM 123 ms / TRT-LLM 194 ms / SGLang 340 ms; SGLang batch=64 460 tok/s.
  ⚠️ Source claims a single H100 runs 70B FP8 (~70 GB weights, at the edge); hardware description is questionable — use with caution.

---

## 5. Long-Context Multi-Node Benchmarks

The only publicly available source for pipeline-parallelism (PP) performance data on long-context prefill. Invaluable for validating the PP modeling and strong-scaling efficiency.

### 5.1 LMSYS Chunked Pipeline Prefill (SGLang): H20 Cluster [LAT]

Source: [lmsys.org blog 2026-01-15 chunked-pipeline](https://www.lmsys.org/blog/2026-01-15-chunked-pipeline/); original charts archived under [`source_charts/`](./source_charts/).
Hardware: 6-node H20 cluster (8×96 GB or 8×141 GB per node); models: Qwen3-235B-A22B-FP8, DeepSeek-V3.1; batch=1, input 128K (also 256K/512K/1M).
Bar groups: `DCK` = adaptive chunk (σ = smoothing factor); numbers = static chunk size (tokens). Values read from charts, ±5%.

#### Qwen3-235B-A22B-FP8, 128K prefill

| Config (GPUs) | Throughput tok/s: DCK / 6144 / 8192 / 12288 / 18432 | TTFT ms: same order |
|---|---|---|
| PP1 TP4 (4) | — / 2,350 / 2,280 / 2,370 / 2,370 | — / 56,000 / 57,800 / 55,600 / 55,500 |
| PP2 TP4 (8) | 4,330 / 4,300 / 4,100 / 4,100 / 3,950 | 31,400 / 31,100 / 32,400 / 32,700 / 35,000 |
| PP4 TP4 (16) | 8,100 / 7,800 / 7,300 / 6,800 / 6,500 | 17,700 / 17,800 / 19,400 / 20,500 / 24,000 |
| PP8 TP4 (32) | 14,600 / 13,250 / 11,800 / 11,400 / 11,100 | 10,500 / 11,000 / 12,800 / 14,400 / 18,500 |

PP8 TP4 DCK long-context extrapolation: 128K → 10.5 s, 256K → 32.7 s, 512K → 114.3 s, 1M → 420.9 s.

#### DeepSeek-V3.1, 128K prefill

| Config (GPUs) | Throughput tok/s: DCK / 4096 / 8192 / 12288 | TTFT ms: same order |
|---|---|---|
| PP1 TP8 (8) | — / 2,450 / 2,600 / 2,650 | — / 53,000 / 50,200 / 48,500 |
| PP1 TP16 (16) | — / 3,800 / 4,350 / 4,450 | — / 34,700 / 30,200 / 29,200 |
| PP2 TP8 (16) | 4,750 / 4,650 / 4,650 / 4,700 | 27,800 / 28,600 / 28,700 / 28,300 |
| PP1 TP32 (32) | — / 5,900 / 6,350 / 6,800 | — / 22,500 / 20,800 / 19,400 |
| PP4 TP8 (32) | 8,800 / 8,550 / 8,050 / 8,000 | 15,500 / 15,800 / 17,000 / 17,800 |

#### Strong Scaling Efficiency (PP=1 as baseline)

| PP degree | Efficiency range | Notes |
|---|---|---|
| PP2 | 0.86–0.91 | Near-linear |
| PP4 | 0.80–0.85 | Qwen DCK = 0.83 |
| PP8 | 0.70–0.77 | Qwen DCK = 0.77, Qwen 6K = 0.70 |

---

## 6. Deployment Configurations

These sources report parallelism layouts without performance numbers. They are kept for validating that the calculator's solver produces layouts consistent with real-world deployments.

### 6.1 DeepSeek [CFG]

- **DeepSeek-V2** ([arXiv 2405.04434](https://arxiv.org/html/2405.04434v1)): 236B MoE / 21B active, single node 8×H800 decode throughput > 50,000 tok/s (large batch) [THR].
- **DeepSeek-V3** ([arXiv 2412.19437](https://arxiv.org/html/2412.19437v1)) §3.4 [CFG]:
  - Prefill units: 4 nodes × 8×H800 = 32 GPUs, TP4 + SP + DP8, EP32
  - Decode units: 40 nodes × 8 = 320 GPUs, TP4, DP80, EP320, dense layers TP1
  - Batch per expert partition ≤ 256 tokens
  - No throughput/latency numbers published

---

## 7. Known Data Gaps

| Gap | Impact | Closest available data |
|---|---|---|
| Qwen3 235B on 8×H100 | Cannot validate MoE + H100 directly | LMSYS H20 32-GPU PP8TP4; Llama 4 Maverick architecture analogy |
| Cross-node EP throughput | EP communication model unvalidated | DeepSeek configs exist but no numbers; Llama 4 Maverick 8×H100 is TP not EP |
| Decode-side PP | PP modeling only validated for prefill | LMSYS §5.1 covers prefill only |
| AMD / Intel GPU | Entirely NVIDIA-focused dataset | No public benchmarks found |
