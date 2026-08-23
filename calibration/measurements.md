# Measured Benchmark Dataset (NVIDIA series)

Collected 2026-08-23. Purpose: compare against the calculator's ideal values to derive calibration factors (see `README.md`).

**Protocol classification** (read before comparing):

- `LAT`: latency protocol, low-concurrency / single-request measurement. Can be compared directly with the calculator's single-batch model.
- `THR`: throughput protocol, continuous-batching saturated steady state. Only usable as an upper bound; do not compare directly.
- `CFG`: deployment config only, no performance numbers.

---

## 1. Single GPU (TP=1, batch=1, llama.cpp) [LAT]

Source: [XiongjieDai/GPU-Benchmarks-on-LLM-Inference](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference).
Models: Llama 3 8B / 70B; `pp` = prefill tok/s, `tg` = decode tok/s; ctx = context length.

| GPU | Model | Quant | ctx | pp (tok/s) | tg (tok/s) |
|---|---|---|---|---|---|
| A100 PCIe 80G | 8B | Q4_K_M | 512 | 5981 | 140.6 |
| A100 PCIe 80G | 8B | F16 | 512 | 7741 | 54.8 |
| A100 PCIe 80G | 70B | Q4_K_M | 512 | 744 | 22.3 |
| A100 SXM 80G | 8B | Q4_K_M | 512 | 5948 | 135.0 |
| A100 SXM 80G | 70B | Q4_K_M | 512 | 818 | 24.6 |
| H100 PCIe 80G | 8B | Q4_K_M | 512 | 8125 | 145.6 |
| H100 PCIe 80G | 8B | F16 | 512 | 10816 | 68.0 |
| H100 PCIe 80G | 8B | F16 | 4096 | 8107 | 66.0 |
| H100 PCIe 80G | 70B | Q4_K_M | 512 | 1013 | 25.0 |
| H100 PCIe 80G | 70B | Q4_K_M | 4096 | 863 | 23.8 |

Note: the repo's **multi-GPU rows** (×2/×4/×8) show abnormally low efficiency (early llama.cpp multi-GPU issues) and are not used for calibration.

## 2. NVIDIA TRT-LLM official overview [THR]

Source: [TensorRT-LLM Performance Overview](https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html).
Metric: Total Output Throughput (tok/s), continuous batching with batch swept to saturation.
Header **ISL/OSL = Input / Output Sequence Length**, i.e. input/output token counts (this project's `workload.inputLen` / `outputLen`); same notation throughout.

### Llama 3.1 8B FP8, 1×H100 (TP=1)

| ISL/OSL | H100 | H200 |
|---|---|---|
| 128/128 | 26401 | 27028 |
| 128/2048 | 21413 | 23102 |
| 1024/2048 | 13166 | 16058 |
| 2048/128 | 3276 | 3391 |
| 2048/2048 | 9462 | 11822 |

### Llama 3.3 70B FP8, 2×H100 (TP=2)

| ISL/OSL | H100 | H200 |
|---|---|---|
| 128/128 | 6092 | 6328 |
| 128/2048 | 5893 | 7467 |
| 1024/2048 | 3785 | 5480 |
| 2048/128 | 723 | 748 |
| 2048/2048 | 2786 | 3776 |

### Llama 3.1 405B FP8, 8×H100 (TP=8)

| ISL/OSL | H100 | H200 |
|---|---|---|
| 128/128 | 3705 | — |
| 128/2048 | 4517 | 4715 |
| 1024/2048 | 3237 | 3610 |
| 2048/128 | 433 | 441 |
| 2048/2048 | 2217 | 2841 |

### Llama 4 Maverick (MoE, 400B / 17B active) FP8, 8×H100 (TP=8)

| ISL/OSL | H100 | H200 |
|---|---|---|
| 128/4096 | 11163 | 18541 |
| 1024/2048 | 11584 | 16859 |
| 2048/128 | 3832 | 4364 |

## 3. NVIDIA official blog: GPT-J 6B, H100 vs A100 [LAT + THR]

Source: [H100 has 4.6x A100 performance in TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/blogs/H100vsA100.html).
TRT-LLM v0.5.0; H100 FP8 / A100 FP16, SXM 80G, TP=1.

| Scenario | H100 | A100 |
|---|---|---|
| batch=1, ISL=128 decode | 185 tok/s (7.1 ms/token) | 111 tok/s (12.5 ms/token) |
| batch=64, ISL/OSL=128/128 | 10907 tok/s, TTFT 102 ms | 3679 tok/s, TTFT 481 ms |

## 4. Splitwise (Microsoft, NSDI'24): Llama-2-70B, TP=8, batch=1 [LAT + PD disaggregation]

Source: [arXiv 2311.18677](https://arxiv.org/html/2311.18677v2).
DGX node, 8 GPUs, TP=8, InfiniBand (A100 pair 200 Gbps / H100 pair 400 Gbps), single request P50.

| Workload | Hardware | TTFT | TBT (=TPOT) | E2E |
|---|---|---|---|---|
| coding (in≈1500 / out=13) | 8×A100 | 185 ms | 52 ms | 856 ms |
| coding (in≈1500 / out=13) | 8×H100 | 95 ms | 31 ms | 493 ms |
| conversation (in≈1020 / out=129) | 8×A100 | 155 ms | 40 ms | 4957 ms |
| conversation (in≈1020 / out=129) | 8×H100 | 84 ms | 28 ms | 3387 ms |

PD-disaggregation findings:
- Async KV-cache transfer adds only **0.8%** to E2E; serialized handoff adds up to +3%
- Prefill-side efficiency degrades once combined input exceeds 2048 tokens
- Decode-side batch saturates memory at 64
- Example optimal splits: coding 35 prefill + 5 decode machines; chat 25 + 15

## 5. DeepSeek [CFG + some THR]

- **DeepSeek-V2** ([arXiv 2405.04434](https://arxiv.org/html/2405.04434v1)): 236B MoE / 21B active, **single node 8×H800 decode throughput > 50,000 tok/s** (large batch) [THR]
- **DeepSeek-V3** ([arXiv 2412.19437](https://arxiv.org/html/2412.19437v1)) §3.4 [CFG]:
  - Prefill units: 4 nodes × 8×H800 = 32 GPUs, TP4 + SP + DP8, EP32
  - Decode units: 40 nodes × 8 = 320 GPUs, TP4, DP80, EP320, dense layers TP1
  - Batch per expert partition ≤ 256 tokens
  - No throughput/latency numbers published

## 6. MLPerf Inference v5.0 (NVIDIA submission) [THR]

Source: [NVIDIA MLPerf v5.0 blog](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0/).
Llama 2 70B, offline (batch saturated):

| System | Throughput (tok/s) |
|---|---|
| 8×H200 | 34,988 |
| 8×B200 | 98,858 |
| GB200 NVL72 | 869,203 |

## 7. Third-party serving (saturated steady state) [THR]

- [dlewis.io: Llama 3.3 70B, 4×H100, NIM/TRT-LLM, bf16](https://dlewis.io/evaluating-llama-33-70b-inference-h100-a100/):
  200→200 peak ~7,000 TPS; 250 concurrent 1000→200 ~2,600 TPS; under load TTFT <5s (queueing-dominated).
- [Cerebrium: Llama 3.1 70B FP8](https://cerebrium.ai/blog/benchmarking-vllm-sglang-tensorrt-for-llama-3-1-api):
  batch=1, in=256 TTFT: vLLM 123ms / TRT-LLM 194ms / SGLang 340ms; SGLang batch=64 460 tok/s.
  ⚠️ The source claims a single H100 runs 70B FP8 (~70GB weights, on the edge); hardware description is questionable — use with caution.

## 8. LMSYS Chunked Pipeline Prefill (SGLang): H20 multi-node, batch=1, 128K long context [PP×TP, read from charts]

Source: [lmsys.org blog 2026-01-15 chunked-pipeline](https://www.lmsys.org/blog/2026-01-15-chunked-pipeline/); original charts archived under [`source_charts/`](./source_charts/).
Hardware: 6-node H20 cluster (8×96GB or 8×141GB per node); models: Qwen3-235B-A22B-FP8, DeepSeek-V3.1; batch=1, input 128K (also 256K/512K/1M).
Bar groups: `DCK` = adaptive chunk (σ = smoothing factor); numbers = static chunk size (tokens). Values read from charts, ±5%.

### Qwen3-235B-A22B-FP8, 128K prefill

| Config (GPUs) | Throughput tok/s: DCK / 6144 / 8192 / 12288 / 18432 | TTFT ms: same order |
|---|---|---|
| PP1 TP4 (4) | — / 2350 / 2280 / 2370 / 2370 | — / 56,000 / 57,800 / 55,600 / 55,500 |
| PP2 TP4 (8) | 4330 / 4300 / 4100 / 4100 / 3950 | 31,400 / 31,100 / 32,400 / 32,700 / 35,000 |
| PP4 TP4 (16) | 8100 / 7800 / 7300 / 6800 / 6500 | 17,700 / 17,800 / 19,400 / 20,500 / 24,000 |
| PP8 TP4 (32) | 14,600 / 13,250 / 11,800 / 11,400 / 11,100 | 10,500 / 11,000 / 12,800 / 14,400 / 18,500 |

PP8 TP4 DCK long-context extrapolation: 128K→10.5s, 256K→32.7s, 512K→114.3s, 1M→420.9s.

### DeepSeek-V3.1, 128K prefill

| Config (GPUs) | Throughput tok/s: DCK / 4096 / 8192 / 12288 | TTFT ms: same order |
|---|---|---|
| PP1 TP8 (8) | — / 2450 / 2600 / 2650 | — / 53,000 / 50,200 / 48,500 |
| PP1 TP16 (16) | — / 3800 / 4350 / 4450 | — / 34,700 / 30,200 / 29,200 |
| PP2 TP8 (16) | 4750 / 4650 / 4650 / 4700 | 27,800 / 28,600 / 28,700 / 28,300 |
| PP1 TP32 (32) | — / 5900 / 6350 / 6800 | — / 22,500 / 20,800 / 19,400 |
| PP4 TP8 (32) | 8800 / 8550 / 8050 / 8000 | 15,500 / 15,800 / 17,000 / 17,800 |

### Strong scaling efficiency (PP=1 as baseline)

PP2 ≈ 0.86–0.91; PP4 ≈ 0.80–0.85 (Qwen DCK 0.83); PP8 ≈ 0.70–0.77 (Qwen DCK 0.77, Qwen 6K 0.70).

## Known data gaps

- **No direct measurement of Qwen3 235B on 8×H100** (closest: LMSYS H20 32-GPU PP8TP4; throughput-architecture analogy: Llama 4 Maverick).
- Cross-node EP throughput measurements exist only as DeepSeek configs, no numbers; Llama 4 Maverick 8×H100 is TP, not EP.
- PP data now available (§8), but only prefill / long-context / batch=1; no decode-side PP measurements.
