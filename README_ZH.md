<div align="center">

# LLM Inference Calculator

**LLM 推理的静态性能模型：给定「模型 + GPU + 互连 + 工作负载」，秒级回答显存够不够、延迟多少、吞吐多少。**

**[English](README.md)**

[**在线体验 →**](https://llm-inference-calculator-delta.vercel.app/)

[![GitHub Stars](https://img.shields.io/github/stars/pochenai/llm-inference-calculator?style=social)](https://github.com/pochenai/llm-inference-calculator)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-blue)](https://www.typescriptlang.org/)
[![Tests](https://img.shields.io/badge/tests-vitest-green)](https://vitest.dev/)

</div>

<p align="center">
  <img src="assets/demo.gif" alt="LLM Inference Calculator Demo" width="100%">
</p>

---

## 功能特性

**开箱即用的硬件与模型库：**

- **403 个模型**：覆盖 Dense 与 MoE 架构，从 SmolLM 135M 到 DeepSeek V3 671B，按参数量分 6 个 tier
- **110+ NVIDIA GPU**：Datacenter（H100/H200/B200/GB200）、DGX、GTX 9/10/16、RTX 20/30/40/50、Pro 全系列
- **5 种 KV Cache 形态**：标准 MHA/GQA、MLA（DeepSeek V3）、线性注意力（Qwen3-Next）、SSM/Mamba（Jamba）、滑动窗口（Gemma 3）
- **互连预设**：NVLink 3/4/5、PCIe、InfiniBand NDR/X800、Spectrum-X 以太网

**两阶段 Roofline 性能模型：**

- **Prefill（算力受限 → TTFT）**：FLOPs 总量 ÷ 有效算力，注意力二次项单独建模
- **Decode（带宽受限 → TPOT/ITL）**：权重 + KV cache 读取 ÷ 有效带宽，MoE 专家覆盖率随 batch 动态变化
- **四条资源轴统一建模**：显存容量 / 显存带宽 / 算力 / 通信，所有开关只改常数不改结构

**全并行策略 + 自动求解：**

- **TP / PP / EP / DP / PD 分离**：统一的「切分 + 通信」建模，TP×EP 协作规则明确（防止常见的重复切分 bug）
- **自动布局求解器**：给定 GPU 数量，自动枚举 TP→PP，优先保证显存可行
- **Speculative Decoding**：draft model 加速，含 draft/main KV 共存显存计算，支持自动推荐 draft model

**纯浏览器运行，零后端成本：**

- 全部计算在浏览器端完成，TypeScript 纯函数核心，与 UI 解耦
- Vite + React UI，一键 `npm run dev` 本地运行

**可视化：**

- **VRAM** 占用拆解（模型 / KV cache / 激活值 / draft model）
- **Prefill** 和 **Decode** 吞吐、延迟、算力利用率和显存利用率
- **Batch Sweep 图**：吞吐量 / 延迟随 batch size 的变化趋势

---

## 快速开始

```bash
git clone https://github.com/pochenai/llm-inference-calculator
cd llm-inference-calculator
npm install
npm run dev  # → http://localhost:5173
```

打开浏览器，选择模型（如 Llama 3.1 70B）、GPU（如 4×H100）、量化精度、batch size，即可看到：

- ✅ 显存是否可行（不可行则反推最大 batch）
- 📊 TTFT（首 token 延迟）、TPOT（每 token 延迟）、端到端延迟、系统吞吐量

---

## 工作原理

整体数据流：

```
(模型, 量化)        --> 三个派生常量: W_bytes, kv_per_token, flops_per_token
(硬件, 并行布局)    --> 每卡容量 / 带宽 / 通信代价
(B, N_in, N_out)    --> 工作负载
        |
        v
  [1] VRAM 检查 (容量轴) --不满足--> 报告 OOM + 反推 B_max
        | 满足
        v
  [2] Prefill (算力轴 + 通信)  --> TTFT
  [3] Decode  (带宽轴 + 通信)  --> TPOT = ITL
        |
        v
  [4] E2E = TTFT + N_out * TPOT
      throughput = B * N_out / E2E
```

**关键简化假设：**

1. **单 batch 一次流过**：Prefill + Decode 严格串行，不建模 continuous batching
2. **量化单一精度**：整个模型量化到同一精度，KV cache 精度可独立设置
3. **流水线 bubble 不建模**：假设 bubble ≈ 0，只计阶段间 P2P 激活搬运
4. **通信模型含延迟项**：`T = msg / BW + α`，decode 小消息场景由 α 主导
5. **理想值模式**：默认输出理论峰值上界，预留 4~5 个效率常数供校准

→ **详细建模推导、公式、校准方法论见 [建模深度解析](docs/modeling_zh.md)**

---

## 校准

默认输出**理想值**（所有效率 = 1），即理论性能上界。预留物理层效率常数供校准：

- `MFU_prefill`：prefill 实际算力利用率
- `BW_eff_decode`：decode 有效显存带宽
- 通信效率（节点内 / 节点间）
- 计算-通信重叠系数

UI 高级面板提供「理想值 / 校准值」一键切换，各参数可手动覆盖。公开实测数据收集在 [`calibration/`](calibration/) 目录。

---

## 对比

| 特性 | 本项目 | 其他在线计算器 |
|---|---|---|
| 建模方法 | 两阶段 Roofline，四轴统一 | 粗估公式或经验值 |
| KV Cache | 5 种形态，含 MLA/SSM/滑动窗口 | 通常只支持标准 MHA |
| 并行策略 | TP/PP/EP/DP/PD 分离 + 自动求解 | 多数只支持 TP |
| 通信建模 | LogP（带宽 + 延迟项 α） | 通常忽略 α |
| 部署 | 纯浏览器，零后端 | 多为在线服务 |
| 开源 | MIT，可审计每一行公式 | 多数闭源 |

---

## 开发

```bash
npm install          # 安装依赖
npm run dev          # UI 开发服务器（http://localhost:5173）
npm run build        # 库构建（dist/，不含 UI）
npm run ui:build     # UI 静态产物（dist-ui/）
npm test             # vitest 单元测试
npm run typecheck    # 全量类型检查
```

**代码结构：**

```
src/
  core/              # 纯函数计算核心（与 UI 解耦，可单测）
    types.ts         # ModelSpec, GpuSpec, ParallelLayout, Workload
    model.ts         # deriveConstants(): W_bytes, KV 几何, FLOPs/token
    memory.ts        # vramBreakdown(): VRAM 分解 + B_max 反推
    latency.ts       # prefillTime() (TTFT), decodeStepTime() (TPOT)
    metrics.ts       # evaluate(): 唯一公开入口，返回 Result<EvaluationResult>
    solver.ts        # solveParallelLayout(): 自动求解 TP/PP
  data/
    models/          # 403 个模型（6B/24B/32B/128B/500B/5000B 分桶）
    gpus/nvidia/     # 110+ GPU（10 个世代）
    network/         # 节点内 / 节点间互连预设
  ui/                # Vite + React 18 + Recharts
```

---

## 引用

如果你的研究或工作中使用了本项目，请按以下方式引用：

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

## 许可证

MIT

---

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐ Star！**

</div>
