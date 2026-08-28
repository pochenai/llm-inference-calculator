# Calibration: Measured vs. Ideal Comparison

This directory bridges the calculator's ideal-value model with real-world measurements. It serves two purposes:

1. **Derive calibration factors**: Compare measured benchmarks against `evaluate()` outputs to anchor efficiency constants
2. **Validate the model**: Check whether ideal values serve as a reasonable upper bound across diverse hardware and workloads

The structured, machine-readable dataset lives in [`../src/data/measurements.ts`](../src/data/measurements.ts). This file documents provenance, comparison methodology, and how to interpret the results.

---

## 1. Protocol Classification

Every benchmark entry carries a protocol tag. **Read this before interpreting any ratio**:

| Protocol | Meaning | Comparison method |
|---|---|---|
| `LAT` | Latency protocol — low-concurrency or single-request measurement | Directly comparable to the single-batch model; ideal values from `evaluate()` |
| `THR` | Throughput protocol — continuous-batching saturated steady state | Compared against the decode roofline ceiling at VRAM-limited max batch; only an upper-bound reference |
| `CFG` | Deployment config only, no performance numbers | Not compared (kept in the archive for layout reference) |

**Why the distinction matters**: The calculator models a single batch flowing through prefill → decode serially. `LAT` benchmarks match this assumption; `THR` benchmarks reflect engines (vLLM, TRT-LLM, SGLang) that sustain throughput via continuous batching — a fundamentally different regime. Comparing `THR` numbers directly against ideal single-batch values would be misleading.

---

## 2. Running the Comparison

```bash
npm run calibrate
```

This renders one row per measurement entry, comparing **measured vs. ideal** for TTFT / TPOT / throughput / E2E. Each column shows `m / i (r)` where:

- `m` = measured value
- `i` = ideal value (from `evaluate()` with current calibration preset)
- `r` = ratio `m / i`

A metric the source did not report, or that does not apply to that protocol, renders as `-`.

---

## 3. Interpreting the Ratios

### LAT rows (directly comparable)

| Metric | Typical ratio range | What it reflects |
|---|---|---|
| **TTFT ratio** | ~1.3–3× | Prefill MFU gap + exposed communication latency (α) |
| **TPOT ratio** | ~2–6× | Decode bandwidth efficiency gap + small-message α (fully exposed at batch=1) |
| **Throughput ratio** | ~0.3–0.7× | Inverse of the above; lower means more headroom for calibration to close |
| **E2E ratio** | ~1.5–5× | Combined effect of TTFT + TPOT gaps |

At batch=1, small-message collectives are fully exposed (α dominates), and prefill MFU is well below 1 — so ratios of 2–5× are the expected calibration region, not a model failure.

### LAT prefill-only rows (LMSYS, `outputLen=1`)

Throughput here means input tok/s (prefill throughput). Ratio < 1 reflects prefill MFU × PP strong-scaling efficiency.

### THR rows (upper-bound only)

Uses `evaluate()` with IDEAL calibration (flash attention enabled, VRAM-limited max batch).

- **Ratio ≈ 1**: The engine achieves the ideal steady-state throughput — unlikely, as it would mean zero overhead
- **Ratio > 1**: The engine outperforms the single-batch model (e.g., continuous batching hides prefill overhead, or the reported batch exceeds our VRAM-limited estimate)
- **Ratio < 1**: The engine is below even the single-batch ideal — indicates the continuous-batching overhead or queuing dominates at the reported configuration

---

## 4. Comparison Results

| id | proto | setup | TTFT m/i (r) | TPOT m/i (r) | Thr m/i (r) | E2E m/i (r) |
|---|---|---|---|---|---|---|
| splitwise-70b-h100-chat | LAT | llama2_70b @ 8×h100 TP8 B2 1020/129 fp16 | 84 / 37 (2.28×) | 28.0 / 5.3 (5.33×) | - | 3,387 / 714 (4.74×) |
| splitwise-70b-h100-code | LAT | llama2_70b @ 8×h100 TP8 B2 1500/13 fp16 | 95 / 55 (1.74×) | 31.0 / 5.3 (5.89×) | - | 493 / 123 (4.01×) |
| splitwise-70b-a100-chat | LAT | llama2_70b @ 8×a100 TP8 B2 1020/129 fp16 | 155 / 117 (1.33×) | 40.0 / 8.6 (4.64×) | - | 4,957 / 1,229 (4.03×) |
| splitwise-70b-a100-code | LAT | llama2_70b @ 8×a100 TP8 B2 1500/13 fp16 | 185 / 173 (1.07×) | 52.0 / 8.6 (6.02×) | - | 856 / 285 (3.00×) |
| gptj-h100-b1 | LAT | gptj_6b @ 1×h100 TP1 B1 128/128 fp8 | - | 7.1 / 1.8 (3.90×) | 185 / 548 (0.34×) | - |
| gptj-h100-b64 | LAT | gptj_6b @ 1×h100 TP1 B64 128/128 fp8 | 102 / 50 (2.03×) | - | 10,907 / 21,034 (0.52×) | - |
| gptj-a100-b1 | LAT | gptj_6b @ 1×a100 TP1 B1 128/128 fp16 | - | 12.5 / 6.0 (2.09×) | 111 / 166 (0.67×) | - |
| gptj-a100-b64 | LAT | gptj_6b @ 1×a100 TP1 B64 128/128 fp16 | 481 / 319 (1.51×) | - | 3,679 / 5,714 (0.64×) | - |
| lmsys-qwen3-235b-pp1tp4 | LAT | qwen3_235b @ 4×h20 TP4 B1 128K/1 fp8 | 56,000 / 27,218 (2.06×) | - | 2,350 / 4,816 (0.49×) | - |
| lmsys-qwen3-235b-pp4tp4 | LAT | qwen3_235b @ 16×h20 TP4PP4 B1 128K/1 fp8 | 17,700 / 6,804 (2.60×) | - | 8,100 / 19,263 (0.42×) | - |
| lmsys-qwen3-235b-pp8tp4 | LAT | qwen3_235b @ 32×h20 TP4PP8 B1 128K/1 fp8 | 10,500 / 3,402 (3.09×) | - | 14,600 / 38,525 (0.38×) | - |
| trtllm-8b-h100-1024-2048 | THR | llama3_1_8b @ 1×h100 TP1 1024/2048 fp8 | - | - | 13,166 / 20,296 (0.65×) | - |
| trtllm-70b-h100-1024-2048 | THR | llama3_3_70b @ 2×h100 TP2 1024/2048 fp8 | - | - | 3,785 / 8,576 (0.44×) | - |
| trtllm-405b-h100-1024-2048 | THR | llama3_405b @ 8×h100 TP8 1024/2048 fp8 | - | - | 3,237 / 10,827 (0.30×) | - |
| trtllm-8b-h100-128-128 | THR | llama3_1_8b @ 1×h100 TP1 128/128 fp8 | - | - | 26,401 / 80,944 (0.33×) | - |
| trtllm-8b-h200-128-128 | THR | llama3_1_8b @ 1×h200 TP1 128/128 fp8 | - | - | 27,028 / 91,546 (0.30×) | - |
| trtllm-8b-h100-128-2048 | THR | llama3_1_8b @ 1×h100 TP1 128/2048 fp8 | - | - | 21,413 / 38,650 (0.55×) | - |
| trtllm-8b-h200-128-2048 | THR | llama3_1_8b @ 1×h200 TP1 128/2048 fp8 | - | - | 23,102 / 57,798 (0.40×) | - |
| trtllm-8b-h200-1024-2048 | THR | llama3_1_8b @ 1×h200 TP1 1024/2048 fp8 | - | - | 16,058 / 29,386 (0.55×) | - |
| trtllm-8b-h100-2048-128 | THR | llama3_1_8b @ 1×h100 TP1 2048/128 fp8 | - | - | 3,276 / 5,419 (0.60×) | - |
| trtllm-8b-h200-2048-128 | THR | llama3_1_8b @ 1×h200 TP1 2048/128 fp8 | - | - | 3,391 / 5,923 (0.57×) | - |
| trtllm-8b-h100-2048-2048 | THR | llama3_1_8b @ 1×h100 TP1 2048/2048 fp8 | - | - | 9,462 / 13,112 (0.72×) | - |
| trtllm-8b-h200-2048-2048 | THR | llama3_1_8b @ 1×h200 TP1 2048/2048 fp8 | - | - | 11,822 / 18,726 (0.63×) | - |
| trtllm-70b-h100-128-128 | THR | llama3_3_70b @ 2×h100 TP2 128/128 fp8 | - | - | 6,092 / 22,230 (0.27×) | - |
| trtllm-70b-h200-128-128 | THR | llama3_3_70b @ 2×h200 TP2 128/128 fp8 | - | - | 6,328 / 24,809 (0.26×) | - |
| trtllm-70b-h100-128-2048 | THR | llama3_3_70b @ 2×h100 TP2 128/2048 fp8 | - | - | 5,893 / 17,368 (0.34×) | - |
| trtllm-70b-h200-128-2048 | THR | llama3_3_70b @ 2×h200 TP2 128/2048 fp8 | - | - | 7,467 / 33,915 (0.22×) | - |
| trtllm-70b-h200-1024-2048 | THR | llama3_3_70b @ 2×h200 TP2 1024/2048 fp8 | - | - | 5,480 / 15,008 (0.37×) | - |
| trtllm-70b-h100-2048-128 | THR | llama3_3_70b @ 2×h100 TP2 2048/128 fp8 | - | - | 723 / 1,441 (0.50×) | - |
| trtllm-70b-h200-2048-128 | THR | llama3_3_70b @ 2×h200 TP2 2048/128 fp8 | - | - | 748 / 1,557 (0.48×) | - |
| trtllm-70b-h100-2048-2048 | THR | llama3_3_70b @ 2×h100 TP2 2048/2048 fp8 | - | - | 2,786 / 5,410 (0.51×) | - |
| trtllm-70b-h200-2048-2048 | THR | llama3_3_70b @ 2×h200 TP2 2048/2048 fp8 | - | - | 3,776 / 9,113 (0.41×) | - |
| trtllm-405b-h100-128-128 | THR | llama3_405b @ 8×h100 TP8 128/128 fp8 | - | - | 3,705 / 17,403 (0.21×) | - |
| trtllm-405b-h100-128-2048 | THR | llama3_405b @ 8×h100 TP8 128/2048 fp8 | - | - | 4,517 / 24,642 (0.18×) | - |
| trtllm-405b-h200-128-2048 | THR | llama3_405b @ 8×h200 TP8 128/2048 fp8 | - | - | 4,715 / 62,164 (0.08×) | - |
| trtllm-405b-h200-1024-2048 | THR | llama3_405b @ 8×h200 TP8 1024/2048 fp8 | - | - | 3,610 / 20,511 (0.18×) | - |
| trtllm-405b-h100-2048-128 | THR | llama3_405b @ 8×h100 TP8 2048/128 fp8 | - | - | 433 / 1,106 (0.39×) | - |
| trtllm-405b-h200-2048-128 | THR | llama3_405b @ 8×h200 TP8 2048/128 fp8 | - | - | 441 / 1,164 (0.38×) | - |
| trtllm-405b-h100-2048-2048 | THR | llama3_405b @ 8×h100 TP8 2048/2048 fp8 | - | - | 2,217 / 6,575 (0.34×) | - |
| trtllm-405b-h200-2048-2048 | THR | llama3_405b @ 8×h200 TP8 2048/2048 fp8 | - | - | 2,841 / 11,548 (0.25×) | - |
| trtllm-maverick-h100-128-4096 | THR | llama4_maverick @ 8×h100 TP8 128/4096 fp8 | - | - | 11,163 / 38,196 (0.29×) | - |
| trtllm-maverick-h200-128-4096 | THR | llama4_maverick @ 8×h200 TP8 128/4096 fp8 | - | - | 18,541 / 107,964 (0.17×) | - |
| trtllm-maverick-h100-1024-2048 | THR | llama4_maverick @ 8×h100 TP8 1024/2048 fp8 | - | - | 11,584 / 38,942 (0.30×) | - |
| trtllm-maverick-h200-1024-2048 | THR | llama4_maverick @ 8×h200 TP8 1024/2048 fp8 | - | - | 16,859 / 102,465 (0.16×) | - |
| trtllm-maverick-h100-2048-128 | THR | llama4_maverick @ 8×h100 TP8 2048/128 fp8 | - | - | 3,832 / 16,196 (0.24×) | - |
| trtllm-maverick-h200-2048-128 | THR | llama4_maverick @ 8×h200 TP8 2048/128 fp8 | - | - | 4,364 / 22,066 (0.20×) | - |
| trtllm-70b-fp4-b200-128-128 | THR | llama3_3_70b @ 1×b200 TP1 128/128 fp4 | - | - | 10,614 / 54,616 (0.19×) | - |
| trtllm-70b-fp4-gb200-128-128 | THR | llama3_3_70b @ 1×gb200 TP1 128/128 fp4 | - | - | 11,101 / 122,336 (0.09×) | - |
| trtllm-70b-fp4-b200-128-2048 | THR | llama3_3_70b @ 1×b200 TP1 128/2048 fp4 | - | - | 9,446 / 60,756 (0.16×) | - |
| trtllm-70b-fp4-gb200-128-2048 | THR | llama3_3_70b @ 1×gb200 TP1 128/2048 fp4 | - | - | 10,276 / 142,570 (0.07×) | - |
| trtllm-70b-fp4-b200-1024-2048 | THR | llama3_3_70b @ 1×b200 TP1 1024/2048 fp4 | - | - | 6,547 / 28,167 (0.23×) | - |
| trtllm-70b-fp4-gb200-1024-2048 | THR | llama3_3_70b @ 1×gb200 TP1 1024/2048 fp4 | - | - | 7,923 / 65,451 (0.12×) | - |
| trtllm-70b-fp4-b200-2048-128 | THR | llama3_3_70b @ 1×b200 TP1 2048/128 fp4 | - | - | 1,330 / 3,461 (0.38×) | - |
| trtllm-70b-fp4-gb200-2048-128 | THR | llama3_3_70b @ 1×gb200 TP1 2048/128 fp4 | - | - | 1,418 / 7,734 (0.18×) | - |
| trtllm-70b-fp4-b200-2048-2048 | THR | llama3_3_70b @ 1×b200 TP1 2048/2048 fp4 | - | - | 4,528 / 17,376 (0.26×) | - |
| trtllm-70b-fp4-gb200-2048-2048 | THR | llama3_3_70b @ 1×gb200 TP1 2048/2048 fp4 | - | - | 5,327 / 40,243 (0.13×) | - |
| trtllm-405b-fp4-b200-128-128 | THR | llama3_405b @ 4×b200 TP4 128/128 fp4 | - | - | 6,219 / 42,218 (0.15×) | - |
| trtllm-405b-fp4-gb200-128-128 | THR | llama3_405b @ 4×gb200 TP4 128/128 fp4 | - | - | 6,599 / - (-) | - |
| trtllm-405b-fp4-b200-128-2048 | THR | llama3_405b @ 4×b200 TP4 128/2048 fp4 | - | - | 7,178 / 119,908 (0.06×) | - |
| trtllm-405b-fp4-gb200-128-2048 | THR | llama3_405b @ 4×gb200 TP4 128/2048 fp4 | - | - | 7,497 / - (-) | - |
| trtllm-405b-fp4-b200-1024-2048 | THR | llama3_405b @ 4×b200 TP4 1024/2048 fp4 | - | - | 4,833 / 42,211 (0.11×) | - |
| trtllm-405b-fp4-gb200-1024-2048 | THR | llama3_405b @ 4×gb200 TP4 1024/2048 fp4 | - | - | 4,686 / - (-) | - |
| trtllm-405b-fp4-b200-2048-128 | THR | llama3_405b @ 4×b200 TP4 2048/128 fp4 | - | - | 738 / 2,630 (0.28×) | - |
| trtllm-405b-fp4-gb200-2048-128 | THR | llama3_405b @ 4×gb200 TP4 2048/128 fp4 | - | - | 762 / - (-) | - |
| trtllm-405b-fp4-b200-2048-2048 | THR | llama3_405b @ 4×b200 TP4 2048/2048 fp4 | - | - | 4,024 / 24,122 (0.17×) | - |
| trtllm-405b-fp4-gb200-2048-2048 | THR | llama3_405b @ 4×gb200 TP4 2048/2048 fp4 | - | - | 4,327 / - (-) | - |
| spheron-70b-h100-fp8 | THR | llama3_3_70b @ 8×h100 TP8 1024/2048 fp8 | - | - | 24,528 / 2,714 (9.04×) | - |
| spheron-70b-h200-fp8 | THR | llama3_3_70b @ 8×h200 TP8 1024/2048 fp8 | - | - | 34,992 / 1,922 (18.20×) | - |
| spheron-70b-b200-fp8 | THR | llama3_3_70b @ 8×b200 TP8 1024/2048 fp8 | - | - | 55,776 / 5,698 (9.79×) | - |
| spheron-70b-b200-fp4 | THR | llama3_3_70b @ 8×b200 TP8 1024/2048 fp4 | - | - | 102,728 / 18,726 (5.49×) | - |
| koyeb-8b-h200-b1-512-512 | LAT | llama3_1_8b @ 1×h200 TP1 B1 512/512 fp16 | - | - | 169 / 297 (0.57×) | 3,160 / 1,726 (1.83×) |
| koyeb-8b-h100-b1-512-512 | LAT | llama3_1_8b @ 1×h100 TP1 B1 512/512 fp16 | - | - | 99 / 207 (0.48×) | 5,390 / 2,469 (2.18×) |
| koyeb-8b-a100-b1-512-512 | LAT | llama3_1_8b @ 1×a100 TP1 B1 512/512 fp16 | - | - | 86 / 126 (0.68×) | 6,160 / 4,070 (1.51×) |
| koyeb-8b-h200-b8-512-512 | LAT | llama3_1_8b @ 1×h200 TP1 B8 512/512 fp16 | - | - | 1,309 / 2,202 (0.59×) | - |
| koyeb-8b-h100-b8-512-512 | LAT | llama3_1_8b @ 1×h100 TP1 B8 512/512 fp16 | - | - | 816 / 1,554 (0.53×) | - |
| koyeb-8b-a100-b8-512-512 | LAT | llama3_1_8b @ 1×a100 TP1 B8 512/512 fp16 | - | - | 652 / 924 (0.71×) | - |
| koyeb-8b-h200-b32-512-512 | LAT | llama3_1_8b @ 1×h200 TP1 B32 512/512 fp16 | - | - | 4,621 / 7,062 (0.65×) | - |
| koyeb-8b-h100-b32-512-512 | LAT | llama3_1_8b @ 1×h100 TP1 B32 512/512 fp16 | - | - | 3,008 / 5,108 (0.59×) | - |
| koyeb-8b-a100-b32-512-512 | LAT | llama3_1_8b @ 1×a100 TP1 B32 512/512 fp16 | - | - | 2,083 / 2,884 (0.72×) | - |
| koyeb-8b-h200-b1-1024-1024 | LAT | llama3_1_8b @ 1×h200 TP1 B1 1024/1024 fp16 | - | - | 168 / 295 (0.57×) | 6,240 / 3,473 (1.80×) |
| koyeb-8b-h100-b1-1024-1024 | LAT | llama3_1_8b @ 1×h100 TP1 B1 1024/1024 fp16 | - | - | 99 / 206 (0.48×) | 10,920 / 4,969 (2.20×) |
| koyeb-8b-a100-b1-1024-1024 | LAT | llama3_1_8b @ 1×a100 TP1 B1 1024/1024 fp16 | - | - | 86 / 125 (0.69×) | 12,290 / 8,191 (1.50×) |
| koyeb-8b-h200-b8-1024-1024 | LAT | llama3_1_8b @ 1×h200 TP1 B8 1024/1024 fp16 | - | - | 1,289 / 2,104 (0.61×) | - |
| koyeb-8b-h100-b8-1024-1024 | LAT | llama3_1_8b @ 1×h100 TP1 B8 1024/1024 fp16 | - | - | 722 / 1,484 (0.49×) | - |
| koyeb-8b-a100-b8-1024-1024 | LAT | llama3_1_8b @ 1×a100 TP1 B8 1024/1024 fp16 | - | - | 632 / 883 (0.72×) | - |
| koyeb-8b-h200-b32-1024-1024 | LAT | llama3_1_8b @ 1×h200 TP1 B32 1024/1024 fp16 | - | - | 4,419 / 6,141 (0.72×) | - |
| koyeb-8b-h100-b32-1024-1024 | LAT | llama3_1_8b @ 1×h100 TP1 B32 1024/1024 fp16 | - | - | 2,401 / 4,423 (0.54×) | - |
| koyeb-8b-a100-b32-1024-1024 | LAT | llama3_1_8b @ 1×a100 TP1 B32 1024/1024 fp16 | - | - | 1,888 / 2,519 (0.75×) | - |
| koyeb-8b-h200-b1-4096-1024 | LAT | llama3_1_8b @ 1×h200 TP1 B1 4096/1024 fp16 | - | - | 164 / 283 (0.58×) | 6,460 / 3,617 (1.79×) |
| koyeb-8b-h100-b1-4096-1024 | LAT | llama3_1_8b @ 1×h100 TP1 B1 4096/1024 fp16 | - | - | 99 / 199 (0.50×) | 10,930 / 5,151 (2.12×) |
| koyeb-8b-a100-b1-4096-1024 | LAT | llama3_1_8b @ 1×a100 TP1 B1 4096/1024 fp16 | - | - | 83 / 119 (0.70×) | 12,690 / 8,577 (1.48×) |
| koyeb-8b-h200-b8-4096-1024 | LAT | llama3_1_8b @ 1×h200 TP1 B8 4096/1024 fp16 | - | - | 1,162 / 1,624 (0.72×) | - |
| koyeb-8b-h100-b8-4096-1024 | LAT | llama3_1_8b @ 1×h100 TP1 B8 4096/1024 fp16 | - | - | 616 / 1,175 (0.52×) | - |
| koyeb-8b-a100-b8-4096-1024 | LAT | llama3_1_8b @ 1×a100 TP1 B8 4096/1024 fp16 | - | - | 544 / 662 (0.82×) | - |
| koyeb-8b-h200-b32-4096-1024 | LAT | llama3_1_8b @ 1×h200 TP1 B32 4096/1024 fp16 | - | - | 3,209 / 3,296 (0.97×) | - |
| koyeb-8b-h100-b32-4096-1024 | LAT | llama3_1_8b @ 1×h100 TP1 B32 4096/1024 fp16 | - | - | 1,591 / 2,482 (0.64×) | - |
| koyeb-8b-a100-b32-4096-1024 | LAT | llama3_1_8b @ 1×a100 TP1 B32 4096/1024 fp16 | - | - | 1,202 / 1,292 (0.93×) | - |
| koyeb-q25-7b-h200-b1-512-512 | LAT | qwen25_7b @ 1×h200 TP1 B1 512/512 fp16 | - | - | 182 / 313 (0.58×) | 2,810 / 1,634 (1.72×) |
| koyeb-q25-7b-h100-b1-512-512 | LAT | qwen25_7b @ 1×h100 TP1 B1 512/512 fp16 | - | - | 105 / 219 (0.48×) | 5,100 / 2,338 (2.18×) |
| koyeb-q25-7b-a100-b1-512-512 | LAT | qwen25_7b @ 1×a100 TP1 B1 512/512 fp16 | - | - | 93 / 133 (0.70×) | 5,740 / 3,853 (1.49×) |
| koyeb-q25-7b-h200-b8-512-512 | LAT | qwen25_7b @ 1×h200 TP1 B8 512/512 fp16 | - | - | 1,371 / 2,378 (0.58×) | - |
| koyeb-q25-7b-h100-b8-512-512 | LAT | qwen25_7b @ 1×h100 TP1 B8 512/512 fp16 | - | - | 808 / 1,678 (0.48×) | - |
| koyeb-q25-7b-a100-b8-512-512 | LAT | qwen25_7b @ 1×a100 TP1 B8 512/512 fp16 | - | - | 699 / 997 (0.70×) | - |
| koyeb-q25-7b-h200-b32-512-512 | LAT | qwen25_7b @ 1×h200 TP1 B32 512/512 fp16 | - | - | 4,523 / 8,083 (0.56×) | - |
| koyeb-q25-7b-h100-b32-512-512 | LAT | qwen25_7b @ 1×h100 TP1 B32 512/512 fp16 | - | - | 2,937 / 5,864 (0.50×) | - |
| koyeb-q25-7b-a100-b32-512-512 | LAT | qwen25_7b @ 1×a100 TP1 B32 512/512 fp16 | - | - | 2,486 / 3,290 (0.76×) | - |
| koyeb-q25-7b-h200-b1-1024-1024 | LAT | qwen25_7b @ 1×h200 TP1 B1 1024/1024 fp16 | - | - | 182 / 312 (0.58×) | 5,620 / 3,278 (1.71×) |
| koyeb-q25-7b-h100-b1-1024-1024 | LAT | qwen25_7b @ 1×h100 TP1 B1 1024/1024 fp16 | - | - | 106 / 218 (0.49×) | 10,040 / 4,689 (2.14×) |
| koyeb-q25-7b-a100-b1-1024-1024 | LAT | qwen25_7b @ 1×a100 TP1 B1 1024/1024 fp16 | - | - | 92 / 132 (0.69×) | 11,410 / 7,729 (1.48×) |
| koyeb-q25-7b-h200-b8-1024-1024 | LAT | qwen25_7b @ 1×h200 TP1 B8 1024/1024 fp16 | - | - | 1,368 / 2,326 (0.59×) | - |
| koyeb-q25-7b-h100-b8-1024-1024 | LAT | qwen25_7b @ 1×h100 TP1 B8 1024/1024 fp16 | - | - | 800 / 1,641 (0.49×) | - |
| koyeb-q25-7b-a100-b8-1024-1024 | LAT | qwen25_7b @ 1×a100 TP1 B8 1024/1024 fp16 | - | - | 682 / 976 (0.70×) | - |
| koyeb-q25-7b-h200-b32-1024-1024 | LAT | qwen25_7b @ 1×h200 TP1 B32 1024/1024 fp16 | - | - | 4,719 / 7,513 (0.63×) | - |
| koyeb-q25-7b-h100-b32-1024-1024 | LAT | qwen25_7b @ 1×h100 TP1 B32 1024/1024 fp16 | - | - | 2,802 / 5,438 (0.52×) | - |
| koyeb-q25-7b-a100-b32-1024-1024 | LAT | qwen25_7b @ 1×a100 TP1 B32 1024/1024 fp16 | - | - | 2,363 / 3,066 (0.77×) | - |
| koyeb-q25-7b-h200-b1-4096-1024 | LAT | qwen25_7b @ 1×h200 TP1 B1 4096/1024 fp16 | - | - | 180 / 304 (0.59×) | 5,690 / 3,369 (1.69×) |
| koyeb-q25-7b-h100-b1-4096-1024 | LAT | qwen25_7b @ 1×h100 TP1 B1 4096/1024 fp16 | - | - | 104 / 213 (0.49×) | 10,200 / 4,797 (2.13×) |
| koyeb-q25-7b-a100-b1-4096-1024 | LAT | qwen25_7b @ 1×a100 TP1 B1 4096/1024 fp16 | - | - | 90 / 128 (0.70×) | 11,860 / 7,987 (1.48×) |
| koyeb-q25-7b-h200-b8-4096-1024 | LAT | qwen25_7b @ 1×h200 TP1 B8 4096/1024 fp16 | - | - | 1,143 / 1,927 (0.59×) | - |
| koyeb-q25-7b-h100-b8-4096-1024 | LAT | qwen25_7b @ 1×h100 TP1 B8 4096/1024 fp16 | - | - | 750 / 1,400 (0.54×) | - |
| koyeb-q25-7b-a100-b8-4096-1024 | LAT | qwen25_7b @ 1×a100 TP1 B8 4096/1024 fp16 | - | - | 636 / 783 (0.81×) | - |
| koyeb-q25-7b-h200-b32-4096-1024 | LAT | qwen25_7b @ 1×h200 TP1 B32 4096/1024 fp16 | - | - | 1,253 / 4,502 (0.28×) | - |
| koyeb-q25-7b-h100-b32-4096-1024 | LAT | qwen25_7b @ 1×h100 TP1 B32 4096/1024 fp16 | - | - | 2,156 / 3,463 (0.62×) | - |
| koyeb-q25-7b-a100-b32-4096-1024 | LAT | qwen25_7b @ 1×a100 TP1 B32 4096/1024 fp16 | - | - | 1,913 / 1,729 (1.11×) | - |

---

## 5. Calibration Anchors

Summary of efficiency constants derived from the comparison above. These are the values used by the "Reset to Calibrated" preset in the UI.

| Constant | Anchor value | Evidence |
|---|---|---|
| `mfuPrefill` | ~0.5 | TRT-LLM batch=64 prefill, SGLang H20 PP=1 prefill (3 cross-checks) |
| `bwEffDecode` | ~0.55 | LAT throughput cross-checks cluster at 0.48–0.57 |
| `commEffIntra` / `commEffInter` | ~0.9 | nccl-tests large-message BW vs peak |
| `tpCommOverlap` / `epCommOverlap` / `ppCommOverlap` | ~0.5 | Conservative start; ideal = 1 (fully overlapped) |
| `alphaIntraMs` | 0.01 ms | NCCL small-message all-reduce total ~6–11 μs |
| `alphaInterMs` | 0.03 ms | IB NDR small-message point-to-point < 1 μs hardware + stack overhead |

---

## 6. Measuring Calibration Constants on Target Hardware

For best accuracy, measure these on the deployment machine rather than relying on the default anchors.

### α (alphaIntraMs / alphaInterMs): Small-Message Collective Latency

**Measure directly with nccl-tests; do NOT derive from inference benchmarks** — batch=1 decode TPOT also contains small-GEMM kernel latency and would contaminate α.

**Intra-node** (= TP-group α):

```bash
# nccl-tests: small-message all-reduce
./build/all_reduce_perf -b 8 -e 8K -g <tp_size>
# Read `avg time` in the 8B–4KB range (bandwidth term negligible → time ≈ α_intra)
```

Without nccl-tests, use a torch.distributed micro-benchmark:

```bash
torchrun --nproc_per_node=8 -c "
import torch, torch.distributed as dist
dist.init_process_group('nccl')
t = torch.zeros(1, device='cuda')
for _ in range(1000): dist.all_reduce(t); torch.cuda.synchronize()
"
```

**Inter-node**: Same command across 2 nodes with `-g 1` → α_inter. For EP all-to-all, use `alltoall_perf` at small sizes.

**Why this is correct**: NCCL auto-selects LL/LL128/tree algorithms at small sizes, so the measured per-call total latency already includes the real algorithm choice — exactly the α semantics of `T_comm = bandwidth_term + α`.

**Sanity ranges**:

| Interconnect | Expected α range |
|---|---|
| NVLink intra-node (8-GPU) | 5–30 μs |
| PCIe intra-node | 50–200 μs |
| IB inter-node | 20–100 μs |

Off by an order of magnitude → check environment (PCIe gen, `NCCL_P2P_LEVEL`, shared NVSwitch).

### Other Constants

| Constant | Measurement method |
|---|---|
| `mfuPrefill` | Measured large-batch prefill tok/s ÷ ideal tok/s |
| `bwEffDecode` | Single-card decode tok/s ÷ (HBM BW ÷ weight bytes per token) |
| `commEffIntra` / `commEffInter` | nccl-tests large message (≥128 MB) measured BW ÷ peak BW |
| `tpCommOverlap` / `epCommOverlap` / `ppCommOverlap` | Fit from exposed-comm scenario (e.g., batch=1 decode measured minus bandwidth term) |

---

## 7. Summary and Known Gaps

The current dataset covers 8 benchmark sources across H100/H200/B200/GB200/A100/H20 hardware, with models from 6B to 405B (Dense and MoE). The calibration anchors are order-of-magnitude estimates (v0) — sufficient for the "Reset to Calibrated" preset but not yet fitted via regression.

**Known gaps**:

- No direct measurement of Qwen3 235B on 8×H100 (closest: LMSYS H20 32-GPU PP8TP4)
- Cross-node EP throughput measurements exist only as DeepSeek deployment configs, no performance numbers
- PP data available (§7 in measurements.md) but only for prefill / long-context / batch=1; no decode-side PP measurements
- Spheron THR data appears to use a prefill-dominated workload (prefillRatio ≈ 1), making throughput ratios > 1 expected and not directly comparable

For the raw benchmark archive with full provenance and per-source notes, see [`measurements.md`](./measurements.md).
