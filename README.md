# LLM Inference Calculator

LLM 推理的静态性能模型。给定一组「模型 + GPU + 互连带宽 + 工作负载」，回答两个问题：

1. **VRAM（GPU 显存）够不够**：给定并行布局，这个模型 + 这个 batch size 能否装下；不够则反推最大 batch size
2. **如果够，延迟和吞吐是多少**：Time To First Token（TTFT，首 token 延迟）、Time Per Output Token（TPOT，每输出 token 耗时，等价于 Inter-Token Latency，ITL）、端到端延迟、系统吞吐量

## 一、简化假设（范围声明）

本项目在以下简化前提下建模，超出范围的场景不做承诺：

1. **不建模 Continuous Batching（连续批处理）与 Paged Attention（分页注意力）**：假设一个 batch 的输入一次性全部送入、一次性计算完成。序列长度 uniform，没有请求的动态进出，也没有 padding 开销。这两项工程优化主要解决显存碎片和动态序列长度问题，在 uniform batch 假设下不影响吞吐结论。
2. **量化单一精度**：整个模型的所有权重量化到同一精度（FP4 / FP8 / FP16 / FP32 之一），用单一的每参数字节数描述。Key-Value（KV）cache 的精度作为独立参数，可选与权重不同。
3. **单 batch 一次流过**：prefill 和 decode 严格串行，不存在两阶段混跑占用的问题（详见附录对旧问题 1 的归档）。
4. **多节点结果不追求精确**：单节点做严格校准，多节点只加一个低阶校正因子（见第八节）。

## 二、建模主脉络：两阶段 + 四条资源轴

LLM 推理唯一的结构性事实：**prefill 和 decode 是物理性质完全相反的两种负载**。所有输入参数都应挂在这两个阶段之下，这是整个模型的主脉络：

| 阶段 | 瓶颈轴 | 物理原因 | 时间主公式 |
|---|---|---|---|
| **Prefill**（决定 TTFT） | 算力（compute bound，计算受限） | 权重只读一次，摊到 `B × N_in` 个 token 上，arithmetic intensity（算术强度）高 | `T_prefill ≈ FLOPs / (MFU × Peak_FLOPs)` |
| **Decode**（决定 TPOT / ITL） | 显存带宽（memory bound，带宽受限） | 每生成 1 个 token 都要重读全部权重 + 全部历史 KV cache，arithmetic intensity ≈ 2 FLOP/byte | `T_step ≈ (W_bytes + KV_read_bytes) / BW_eff` |

四条资源轴：

1. **显存容量** —— 决定「能不能跑」（VRAM 检查）
2. **显存带宽** —— 决定 decode 快不快
3. **算力 FLOPs** —— 决定 prefill 快不快
4. **通信带宽** —— 并行引入的第四条轴

关键设计原则：**所有场景开关（FlashAttention、量化、Prefill-Decode 分离即 PD 分离、各种并行）都不改变模型结构，只改变这四条轴上的某个数值**。Roofline 四区域的几何解释见 `roofline-regions.svg`。

整体数据流：

```
(model, quant)      --> 三个派生常量: W_bytes, kv_per_token, flops_per_token
(hardware, layout)  --> 每卡容量 / 带宽 / 通信代价
(B, N_in, N_out)    --> 工作负载
        |
        v
  [1] VRAM 检查 (容量轴) --不满足--> 报告 OOM + 反推 B_max
        | 满足
        v
  [2] T_prefill (算力轴 + 通信)  --> TTFT
  [3] T_step    (带宽轴 + 通信)  --> TPOT = ITL
        |
        v
  [4] E2E = TTFT + N_out * TPOT
      throughput = B * N_out / E2E
```

## 三、Step 0：把模型压缩成三个派生常量

输入模型规格：层数 `L`、隐藏维度 `h`、注意力头类型（Multi-Head Attention，MHA / Grouped Query Attention，GQA 的 `n_heads` 与 `n_kv_heads`）、每头维度 `d_head`、FFN 比例、Mixture of Experts（MoE，混合专家）配置、词表大小。

只派生三个常量，后续全部复用：

```
W_bytes         = P_total * b_w                # total weight bytes; b_w = bytes per param under quantization
kv_per_token    = 2 * L * n_kv_heads * d_head * b_kv   # KV cache bytes per token (2 = K and V); GQA takes effect here
flops_per_token ~= 2 * P_active                # matmul FLOPs per token; MoE uses active params only
```

Attention（注意力）的二次项单独记：prefill 阶段每层注意力约 `4 * N^2 * d_model` FLOPs，这是唯一与序列长度平方相关的计算量。

## 四、Step 1：VRAM 模型（容量轴）

每卡显存占用：

```
KV_total       = kv_per_token * (N_in + N_out) * B
VRAM_per_gpu   = (W_bytes + KV_total + Activation) / shard + overhead
```

- `shard` 由并行布局决定（见第五节的切分规则）
- **Activation（激活值）是 FlashAttention 开关唯一的落点**：
  - 开 FlashAttention：`O(B * N * h)`，随序列长度线性增长
  - 关 FlashAttention：注意力分数矩阵 `O(B * n_heads * N^2)`，长序列出现内存悬崖，直接 Out Of Memory（OOM，显存溢出）

判定与反推：

```
feasible = VRAM_per_gpu <= VRAM_capacity * (1 - headroom)
B_max    = 解出 VRAM 约束关于 B 的上界    # reported when infeasible
```

## 五、Step 2：并行 = 切分算子 + 通信项

Tensor Parallelism（TP，张量并行）、Pipeline Parallelism（PP，流水线并行）、Expert Parallelism（EP，专家并行）、Data Parallelism（DP，数据并行）不各建一套模型，它们是同一个模式的不同实例：**切分某类显存占用 + 添加一项通信代价**。

| 并行方式 | 切分什么（显存收益） | 添加什么通信（代价） | 适用场景 |
|---|---|---|---|
| **TP**（size = t） | 权重、KV cache、激活 ÷ t | 每层 2 次 all-reduce，消息量 ∝ `B × N × h` | 仅限节点内（吃高带宽）；prefill / 大 batch 收益大 |
| **PP**（size = p） | 权重、KV cache ÷ p（按层切） | 阶段间点对点传激活，量小；但有 bubble（空洞）`(p-1)/m`（m = microbatch 数） | 跨节点便宜；m 小时 bubble 伤延迟 |
| **EP**（size = e） | 专家权重 ÷ e | 每个 MoE 层 2 次 all-to-all（dispatch + combine，分发 + 合并） | 仅 MoE 模型 |
| **DP**（size = d） | 不切分（权重全量复制） | 推理期无通信 | 纯吞吐 × d |
| **PD 分离** | 按阶段切分，不按模型切分 | KV cache 从 prefill 池搬到 decode 池，耗时 `KV_total / BW_节点间` | 让两个池各自选最优并行与硬件配置 |

约束方程：

```
TP * PP * DP = N_gpu
TP group must fit within one node      # TP communication demands intra-node bandwidth
```

## 六、Step 3：延迟模型

**TTFT（prefill，算力轴）：**

```
FLOPs_prefill = B * N_in * flops_per_token + 4 * L * N_in^2 * d_model * B
T_compute     = FLOPs_prefill / (N_gpu * Peak_FLOPs * MFU)     # MFU = Model FLOPs Utilization
T_comm        = TP all-reduce per layer + PP bubble
TTFT          = T_compute + non_overlapped_comm (+ KV transfer if PD-disaggregated)
```

**TPOT（decode，带宽轴）：**

```
bytes_per_step = W_bytes / (t * p) + kv_per_token / (t * p) * S_history * B
T_step         = bytes_per_step / (BW_eff * group_size) + T_tp_comm
TPOT = ITL     = T_step
```

注意：decode 阶段 TP 的 all-reduce 消息很小（`B × h`），此时通信是延迟受限而非带宽受限——这正是「TP 在小 batch 下反而亏」的来源。而 TP 对 decode 的收益在于聚合多卡 HBM 带宽，两头对冲后是否划算由模型自动算出。

## 七、Step 4：指标汇总

```
E2E_latency  = TTFT + N_out * TPOT
throughput   = B * N_out / E2E_latency      # output tokens per second, system level
```

单 batch 一次流过的前提下，prefill 和 decode 串行相加，不存在「取最慢」或「并发」的分支。

微观 / 宏观两个视角必须区分，否则建模时容易概念混淆：

- **微观视角（Per-Request Latency，单请求延迟）**：一个请求从进来到离开卡了多久。核心指标是 TTFT 和 TPOT，对应用户体验与服务等级协议（Service Level Agreement，SLA）。
- **宏观视角（System Throughput，系统吞吐）**：这组 GPU 每秒产出多少有效 token。核心指标是 output tokens/sec，对应硬件利用率和成本效益。

## 八、场景开关如何进入模型

再次强调：开关只改常数，不改结构。

| 开关 | 影响哪条轴 | 模型内的体现 |
|---|---|---|
| FlashAttention 开 / 关 | 容量轴 + prefill | 关：激活出现 `N^2` 项；长序列注意力转为带宽受限 |
| 量化精度 | 容量轴 + 两阶段时间 | `b_w` 整体缩放 `W_bytes`；KV cache 量化由独立的 `b_kv` 控制 |
| PD 分离 | 通信轴 | TTFT 加 KV cache 传输项；prefill / decode 池使用各自独立的并行布局 |
| MoE | 派生常量 | `W_bytes` 含全部专家，`flops_per_token` 用激活参数；EP 变为可选 |

## 九、校准策略

整个模型只有 4~5 个自由效率常数：

- `MFU_prefill`：prefill 阶段实际算力利用率
- `BW_eff_decode`：decode 阶段有效显存带宽
- 通信效率（节点内 / 节点间各一个）
- 计算-通信重叠系数

单节点用实测数据拟合这几个常数即可，参数空间小，校准是良态问题。多节点不追求精确，只额外引入一个关于通信量 / 跳数的校正因子，形式限于线性、最多二次模型（与第一节简化假设 4 对应）。

## 十、技术方案与代码结构

- **语言**：TypeScript，计算核心为纯函数库，与 UI 解耦，可单测
- **部署**：Vercel。全部计算在浏览器端完成，无后端、无运行时成本
- **框架**：Vite + React（或 Next.js 静态导出），待定；计算核心不依赖所选框架

```
src/
  model.ts        # ModelSpec -> derived constants (W_bytes, kv_per_token, flops_per_token)
  hardware.ts     # GpuSpec (VRAM / FLOPs / bandwidth), Interconnect (intra-node / inter-node)
  layout.ts       # ParallelLayout (TP, PP, DP, EP, PD disaggregation)
  memory.ts       # vramBreakdown() -> per-GPU breakdown + feasibility + B_max
  latency.ts      # ttft(), tpot()
  metrics.ts      # e2eLatency(), throughput()
  calibration.ts  # efficiency constants (single-node calibrated)
  ui/             # input form + result view (framework-dependent)
```

## 附录：旧问题归档

**问题 1：prefill 和 decode 同时存在时，GPU 吞吐怎么算？取最慢的？流水线？还是并发？**
归档结论：在「单 batch 一次流过」的简化假设下，两阶段严格串行，`E2E = TTFT + N_out × TPOT`。「取 min」「混跑并发」只存在于 continuous batching 系统（如 SGLang / vLLM 的运行时调度）中，不在本项目范围内。

**问题 2：SGLang 的 benchmark 吞吐是直接算出来的吗？**
归档结论：SGLang 报告的吞吐来自 continuous batching 下大量请求混跑、持续填满硬件的稳态测量，与本项目的单 batch 串行模型前提不同，数字不可直接对比；引用时必须标注前提差异。
