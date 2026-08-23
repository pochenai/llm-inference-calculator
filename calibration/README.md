# Benchmark Data & Formal Comparison (NVIDIA series)

The **structured, machine-readable** dataset lives in [`../src/data/measurements.ts`](../src/data/measurements.ts).
Each entry carries `protocol`, `sourceUrl`, the full scenario (model / gpu / layout /
quant / batch / ISL / OSL) and only the metrics the source actually reported.
This file (and [`measurements.md`](./measurements.md)), the human-readable archive documents provenance and how to interpret the comparison.

## Run the comparison

```
npm run calibrate
```

This renders one row per entry, comparing **measured vs ideal** for
**TTFT / TPOT / throughput / E2E** together. 

A metric the source did not report, or
that does not apply to that protocol, renders as `-`. Columns are `m / i (r)` where
`m` = measured, `i` = ideal, `r` = measured/ideal ratio.


### Results

| id                         | proto | setup                                              | TTFT m/i (r)            | TPOT m/i (r)        | Thr m/i (r)             | E2E m/i (r)           |
|----------------------------|-------|----------------------------------------------------|-------------------------|---------------------|-------------------------|-----------------------|
| splitwise-70b-h100-chat    | LAT   | llama2_70b @ 8xh100_sxm TP8PP1 B1 1020/129 fp16    | 84 / 25 (3.33x)         | 28.0 / 6.8 (4.09x)  | - / 142 (-)             | 3,387 / 908 (3.73x)   |
| splitwise-70b-h100-code    | LAT   | llama2_70b @ 8xh100_sxm TP8PP1 B1 1500/13 fp16     | 95 / 37 (2.60x)         | 31.0 / 6.8 (4.53x)  | - / 104 (-)             | 493 / 126 (3.93x)     |
| splitwise-70b-a100-chat    | LAT   | llama2_70b @ 8xa100_sxm TP8PP1 B1 1020/129 fp16    | 155 / 68 (2.29x)        | 40.0 / 10.4 (3.85x) | - / 92 (-)              | 4,957 / 1,407 (3.52x) |
| splitwise-70b-a100-code    | LAT   | llama2_70b @ 8xa100_sxm TP8PP1 B1 1500/13 fp16     | 185 / 100 (1.86x)       | 52.0 / 10.4 (5.01x) | - / 55 (-)              | 856 / 235 (3.65x)     |
| gptj-h100-b1               | LAT   | gptj_6b @ 1xh100_sxm TP1PP1 B1 128/128 fp8         | - / 1 (-)               | 7.1 / 1.8 (3.90x)   | 185 / 548 (0.34x)       | - / 234 (-)           |
| gptj-h100-b64              | LAT   | gptj_6b @ 1xh100_sxm TP1PP1 B64 128/128 fp8        | 102 / 50 (2.03x)        | - / 2.6 (-)         | 10,907 / 21,034 (0.52x) | - / 389 (-)           |
| gptj-a100-b1               | LAT   | gptj_6b @ 1xa100_sxm TP1PP1 B1 128/128 fp16        | - / 5 (-)               | 12.5 / 6.1 (2.05x)  | 111 / 163 (0.68x)       | - / 785 (-)           |
| gptj-a100-b64              | LAT   | gptj_6b @ 1xa100_sxm TP1PP1 B64 128/128 fp16       | 481 / 319 (1.51x)       | - / 8.9 (-)         | 3,679 / 5,629 (0.65x)   | - / 1,455 (-)         |
| llamacpp-h100-8b-f16       | LAT   | llama3_1_8b @ 1xh100_pcie TP1PP1 B1 512/512 fp16   | 47 / 11 (4.28x)         | 14.7 / 8.1 (1.82x)  | - / 123 (-)             | - / 4,148 (-)         |
| llamacpp-h100-70b-q4       | LAT   | llama3_3_70b @ 1xh100_pcie TP1PP1 B1 512/512 int4  | 505 / 24 (20.94x)       | 40.0 / 17.8 (2.25x) | - / 56 (-)              | - / 9,125 (-)         |
| llamacpp-a100-70b-q4       | LAT   | llama3_3_70b @ 1xa100_sxm TP1PP1 B1 512/512 int4   | 626 / 58 (10.70x)       | 40.7 / 17.8 (2.29x) | - / 56 (-)              | - / 9,160 (-)         |
| lmsys-qwen3-235b-pp1tp4    | LAT   | qwen3_235b @ 4xh20 TP4PP1 B1 131072/1 fp8          | 56,000 / 27,218 (2.06x) | - / - (-)           | 2,350 / 4,816 (0.49x)   | - / - (-)             |
| lmsys-qwen3-235b-pp4tp4    | LAT   | qwen3_235b @ 16xh20 TP4PP4 B1 131072/1 fp8         | 17,700 / 6,804 (2.60x)  | - / - (-)           | 8,100 / 19,263 (0.42x)  | - / - (-)             |
| lmsys-qwen3-235b-pp8tp4    | LAT   | qwen3_235b @ 32xh20 TP4PP8 B1 131072/1 fp8         | 10,500 / 3,402 (3.09x)  | - / - (-)           | 14,600 / 38,525 (0.38x) | - / - (-)             |
| trtllm-8b-h100-1024-2048   | THR   | llama3_1_8b @ 1xh100_sxm TP1PP1 B0 1024/2048 fp8   | - / - (-)               | - / - (-)           | 13,166 / 22,173 (0.59x) | - / - (-)             |
| trtllm-70b-h100-1024-2048  | THR   | llama3_3_70b @ 2xh100_sxm TP2PP1 B0 1024/2048 fp8  | - / - (-)               | - / - (-)           | 3,785 / 10,161 (0.37x)  | - / - (-)             |
| trtllm-405b-h100-1024-2048 | THR   | llama3_1_405b @ 8xh100_sxm TP8PP1 B0 1024/2048 fp8 | - / - (-)               | - / - (-)           | 3,237 / 15,036 (0.22x)  | - / - (-)             |

## Protocol classification (read before interpreting)

| Protocol | Meaning | How it is compared |
|---|---|---|
| `LAT` | Latency protocol, low-concurrency / single-request | Directly comparable to the single-batch model; ideal from `evaluate()` |
| `THR` | Throughput protocol, continuous-batching saturated steady state | Compared against the **decode roofline ceiling** at the VRAM-limited max batch; only an upper bound |
| `CFG` | Deployment config only, no perf numbers | Not compared (kept in the archive for layout sanity) |

## Interpreting the ratios

- **LAT rows**: ratio ~2–5× is the expected calibration region — at batch=1 the
  small-message collectives are fully exposed (α) and prefill MFU is well below 1.
  The TTFT ratio ≈ prefill-MFU gap; the TPOT ratio ≈ decode efficiency + exposed α.
- **LAT prefill-only rows** (LMSYS, `outputLen=1`): throughput = input tok/s; ratio <1
  reflects prefill MFU × PP strong-scaling efficiency.
- **THR rows**: ratio <1 = fraction of the decode roofline ceiling achieved by the
  saturated engine. It decreases for larger models (8B ≈ 0.59, 70B ≈ 0.37, 405B ≈ 0.22)
  because weight reads amortize less and TP communication grows.
- **Outliers**: `llamacpp-*-q4` prefill ratios (~11–21×) are inflated because the
  ideal uses int4/FP4 tensor-core peak that llama.cpp's Q4 kernels do not reach;
  treat those as decode-bandwidth anchors (TPOT ratio ~2.2×), not prefill anchors.

## Calibration anchors (v0, order-of-magnitude only)

| Constant | Anchor | Evidence |
|---|---|---|
| `bwEffDecode` | ~0.53 | llama.cpp single-card decode ×3; throughput cross-checks 0.48–0.57 |
| `mfuPrefill` | ~0.17 (llama.cpp) / ~0.5 (TRT-LLM batch=64, SGLang H20 PP=1) | prefill comparisons ×3 |
| `ppEfficiency` | ~0.8–0.9 (chunked prefill, PP≤8) | LMSYS strong scaling 0.77–0.91 |
| α (small-msg collective latency) | default intra 0.01 ms / inter 0.03 ms; measure on target (below) | Splitwise batch=1 ratios 4–6 indirectly confirm its presence |

## Measuring calibration constants on the target machine

Later the UI will expose a `Calibration` input box; per-constant measurement:

### α (`alphaIntraMs` / `alphaInterMs`): per-call small-message collective latency

**Measure directly; do not back it out of inference data** — batch=1 decode TBT also
contains small-GEMM kernel latency and would contaminate α.

1. **Intra-node** (= TP-group α): nccl-tests small messages,
   `./build/all_reduce_perf -b 8 -e 8K -g <tp>`; read the `avg time` in the 8B–4KB
   range (bandwidth term negligible there, so the time ≈ α_intra).
   Without nccl-tests: a torch.distributed micro-benchmark (`torchrun --nproc_per_node=8`,
   loop `dist.all_reduce` on a small tensor, warm up, average ~1000 iters with cuda sync).
2. **Inter-node**: same command across 2 nodes with `-g 1` → α_inter. For EP all-to-all,
   use `alltoall_perf` at small sizes (the model currently reuses α_intra; may split later).
3. **Why this is correct**: NCCL auto-selects LL/LL128/tree at small sizes, so the
   measured per-call total latency already includes the real algorithm choice — exactly
   the α semantics of `ringAllreduceMs = bandwidth term + α` (per-call, not per-hop).
4. **Sanity ranges**: NVLink intra-node 8-GPU ≈ 5–30 μs; PCIe ≈ 50–200 μs; IB inter-node
   ≈ 20–100 μs. Off by an order of magnitude ⇒ check the environment (PCIe gen,
   NCCL_P2P_LEVEL, shared NVSwitch). Fill in ms (divide a μs reading by 1000).
5. **Defaults when unmeasured**: `DEFAULT_ALPHA_INTRA_MS = 0.01`, `DEFAULT_ALPHA_INTER_MS = 0.03`,
   same order as the tps old heuristic (8/25 μs per hop) and measured NCCL small-message
   totals (6–11 μs). Note α only takes effect when the matching `*CommOverlap < 1`
   (decode small-batch collectives are not overlapped). Do not derive α from batch=1 TBT.

### Other constants

| Constant | How to measure |
|---|---|
| `mfuPrefill` | measured large-batch prefill tok/s ÷ ideal tok/s |
| `bwEffDecode` | single-card decode tok/s ÷ (BW / weight bytes) |
| `commEffIntra/Inter` | nccl-tests **large** message (≥128MB) measured BW ÷ peak BW |
| `tpCommOverlap` / `epCommOverlap` / `ppCommOverlap` | fit from an exposed-comm scenario (e.g. batch=1 decode measured minus bandwidth term); ideal = 1, conservative start ≈ 0.5 |
| `ppEfficiency` (not yet a Calibration field) | LMSYS anchor 0.8–0.9 |
