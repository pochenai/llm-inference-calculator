<div align="center">

# LLM Inference Calculator

**A static performance model for LLM inference: given (model, GPU, interconnect, workload), instantly answer VRAM feasibility, latency, and throughput.**

**[中文](README_ZH.md)**

[**Try it online →**](https://llm-inference-calculator-delta.vercel.app/)

[![GitHub Stars](https://img.shields.io/github/stars/pochenai/llm-inference-calculator?style=social)](https://github.com/pochenai/llm-inference-calculator)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-blue)](https://www.typescriptlang.org/)
[![Tests](https://img.shields.io/badge/tests-vitest-green)](https://vitest.dev/)

</div>

<p align="center">
  <img src="assets/demo.gif" alt="LLM Inference Calculator Demo" width="100%">
</p>

---

## Features

**Ready-to-use hardware and model libraries:**

- **403 models**: Dense and MoE architectures, from SmolLM 135M to DeepSeek V3 671B, organized in 6 parameter tiers
- **110+ NVIDIA GPUs**: Datacenter (H100/H200/B200/GB200), DGX, GTX 9/10/16, RTX 20/30/40/50, and full Pro lineup
- **5 KV cache variants**: Standard MHA/GQA, MLA (DeepSeek V3), Linear Attention (Qwen3-Next), SSM/Mamba (Jamba), Sliding Window (Gemma 3)
- **Interconnect presets**: NVLink 3/4/5, PCIe, InfiniBand NDR/X800, Spectrum-X Ethernet

**Two-phase Roofline performance model:**

- **Prefill (compute bound → TTFT)**: Total FLOPs ÷ effective compute, with attention quadratic term modeled separately
- **Decode (bandwidth bound → TPOT/ITL)**: Weight + KV cache reads ÷ effective bandwidth, with MoE expert coverage varying dynamically with batch size
- **Four resource axes unified**: VRAM capacity / memory bandwidth / compute / communication — all switches change constants, not structure

**Full parallelism support + automatic solver:**

- **TP / PP / EP / DP / PD disaggregation**: Unified "shard + communicate" model, explicit TP×EP interaction rules (preventing common double-sharding bugs)
- **Automatic layout solver**: Given GPU count, enumerates TP→PP, prioritizing VRAM feasibility
- **Speculative Decoding**: Draft model acceleration with draft/main KV coexistence memory calculation, supports automatic draft model recommendation

**Runs entirely in the browser, zero backend cost:**

- All computation runs client-side, TypeScript pure-function core decoupled from UI
- Vite + React UI, one `npm run dev` to run locally

**Visualization:**

- **VRAM** breakdown (model / KV cache / activation / draft model)
- **Prefill** and **Decode** throughput, latency, compute utilization, and memory utilization
- **Batch Sweep chart**: Throughput and latency trends across batch sizes

---

## Quick Start

```bash
git clone https://github.com/pochenai/llm-inference-calculator
cd llm-inference-calculator
npm install
npm run dev  # → http://localhost:5173
```

Open your browser, select a model (e.g. Llama 3.1 70B), GPUs (e.g. 4×H100), quantization, and batch size to see:

- ✅ VRAM feasibility (if infeasible, the maximum batch size is computed)
- 📊 TTFT (time to first token), TPOT (time per output token), end-to-end latency, system throughput

---

## How It Works

Overall data flow:

```
(model, quant)      --> 3 derived constants: W_bytes, kv_per_token, flops_per_token
(hardware, layout)  --> per-GPU capacity / bandwidth / communication cost
(B, N_in, N_out)    --> workload
        |
        v
  [1] VRAM check (capacity) --infeasible--> report OOM + back-solve B_max
        | feasible
        v
  [2] Prefill (compute + communication)  --> TTFT
  [3] Decode  (bandwidth + communication) --> TPOT = ITL
        |
        v
  [4] E2E = TTFT + N_out * TPOT
      throughput = B * N_out / E2E
```

**Key simplifying assumptions:**

1. **Single batch, one pass**: Prefill + Decode are strictly sequential; continuous batching is not modeled
2. **Uniform quantization**: The entire model is quantized to the same precision; KV cache precision can be set independently
3. **Pipeline bubbles not modeled**: Assumes bubble ≈ 0, only inter-stage P2P activation transfer is counted
4. **Communication model includes latency term**: `T = msg / BW + α`, decode small-message scenarios are dominated by α
5. **Ideal-value mode**: Outputs theoretical peak upper bound by default, with 4–5 efficiency constants available for calibration

→ **Detailed modeling derivation, formulas, and calibration methodology: [Modeling Deep Dive](docs/modeling.md)**

---

## Calibration

Outputs **ideal values** by default (all efficiencies = 1), representing theoretical performance upper bounds. Physical-layer efficiency constants are available for calibration:

- `MFU_prefill`: Actual compute utilization during prefill
- `BW_eff_decode`: Effective memory bandwidth during decode
- Communication efficiency (intra-node / inter-node)
- Compute-communication overlap coefficient

The UI advanced panel provides one-click switching between "ideal" and "calibrated" presets; each parameter can be manually overridden. Publicly available measurement data is collected in the [`calibration/`](calibration/) directory.

---

## Comparison

| Feature | This Project | Other Online Calculators |
|---|---|---|
| Modeling approach | Two-phase Roofline, 4 axes unified | Rough formulas or empirical values |
| KV cache | 5 variants, including MLA/SSM/sliding window | Usually only standard MHA |
| Parallelism | TP/PP/EP/DP/PD disaggregation + auto solver | Most support only TP |
| Communication | LogP (bandwidth + latency term α) | Usually ignores α |
| Deployment | Browser-only, zero backend | Mostly online services |
| Open source | MIT, every formula line is auditable | Mostly closed source |

---

## Development

```bash
npm install          # Install dependencies
npm run dev          # UI dev server (http://localhost:5173)
npm run build        # Library build (dist/, excludes UI)
npm run ui:build     # UI static build (dist-ui/)
npm test             # vitest unit tests
npm run typecheck    # Full type check
```

**Code structure:**

```
src/
  core/              # Pure-function compute core (UI-decoupled, unit-testable)
    types.ts         # ModelSpec, GpuSpec, ParallelLayout, Workload
    model.ts         # deriveConstants(): W_bytes, KV geometry, FLOPs/token
    memory.ts        # vramBreakdown(): VRAM decomposition + B_max back-solve
    latency.ts       # prefillTime() (TTFT), decodeStepTime() (TPOT)
    metrics.ts       # evaluate(): single public entry, returns Result<EvaluationResult>
    solver.ts        # solveParallelLayout(): auto-solve TP/PP
  data/
    models/          # 403 models (6B/24B/32B/128B/500B/5000B tiers)
    gpus/nvidia/     # 110+ GPUs (10 generations)
    network/         # Intra-node / inter-node interconnect presets
  ui/                # Vite + React 18 + Recharts
```

---

## Citation

If you use this project in your research or work, please cite it as:

```bibtex
@misc{llm-inference-calculator,
  author       = {Po Chen},
  title        = {LLM Inference Calculator: A Static Performance Model for LLM Inference},
  howpublished = {\url{https://github.com/pochenai/llm-inference-calculator}},
  note         = {Accessed: 08/2026},
  year         = {08/2026}
}
```

---

## License

MIT

---

<div align="center">

**If this project helps you, please consider giving it a ⭐ Star!**

</div>
