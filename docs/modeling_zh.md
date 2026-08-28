# LLM 推理性能建模：从零推导 TTFT / TPOT / 吞吐量，附网页版计算器

## 1. Motivation

本地部署 LLM，或者公司做 LLM 容量规划时，绕不开两个问题：

1. **显存够不够用**：用什么卡、多少张卡、能装多大的 batch size、并行策略怎么选？
2. **如果够，延迟和吞吐是多少**：Time To First Token（TTFT，首 token 延迟）、Time Per Output Token（TPOT，每输出 token 耗时，等价于 Inter-Token Latency / ITL）、端到端延迟、系统吞吐量。

问题在于，延迟和吞吐涉及的变量太多了——batch size、输入长度、输出长度、prefill 和 decode 的比例等参数全是动态的，不同并行策略之间的交叉影响更是一团乱麻。现有的估算工具（如 [vram-calculator](https://apxml.com/tools/vram-calculator)、[tps](https://tps.bunai.cc/) 等），要么支持的参数范围太窄，要么引入了过多复杂假设，导致粗估结果不够准确、也很难验证。

所以我们换一个思路：**用尽可能简化的假设，计算 LLM 的静态理想推理性能——给定模型和硬件配置，快速得到吞吐和延迟的理论上界**。同时配套了一个网页版计算器，可以直接上手体验。

> 对计算原理不感兴趣的读者，可以直接跳到[网页版计算器](https://llm-inference-calculator-delta.vercel.app/)，先感受一下效果。


## Terminology

后文涉及不少符号和缩略词，这里先集中定义，方便随时回查。也可直接跳到[下一节]()。

**工作负载与模型规格**

| 符号 | 含义 |
|---|---|
| `B` | batch size（同时处理的请求数） |
| `N_in` / `N_out` | 输入 / 输出序列长度（token 数） |
| `S` | 序列长度（含历史 token），decode 阶段 `S = N_in + 已生成 token 数` |
| `h` | 隐藏维度（hidden size，残差流宽度） |
| `L` | 模型层数 |
| `P_total` / `P_active` | 总参数量 / 每 token 激活参数量（dense 模型两者相等；MoE 模型 `P_active < P_total`） |
| `q_heads` / `kv_heads` | 查询注意力头数 / KV 注意力头数（GQA 中 `kv_heads < q_heads`） |
| `head_dim` | 每头维度 |
| `qDim` | 查询投影维度 `= q_heads × head_dim`。标准 MHA 中 `qDim = h`；MLA 中可能 `< h` |
| `r` / `prefillRatio` | prefill 请求占比（0~1），非 PD 模式下用于估算两阶段的 GPU 时间分配 |
| `ρ_prefill` / `ρ_decode` | prefill / decode 各占 GPU 时间的比例（由 `r` 和各阶段吞吐率推算，`ρ_prefill + ρ_decode = 1`） |

**并行维度**（`TP × EP × PP × DP = N_gpu`）

| 符号 | 含义 |
|---|---|
| `t` / `TP` | Tensor Parallelism（张量并行）大小——层内切分权重，仅限节点内 |
| `p` / `PP` | Pipeline Parallelism（流水线并行）大小——按层切分，可跨节点 |
| `e` / `EP` | Expert Parallelism（专家并行）大小——分发专家，仅 MoE 模型 |
| `d` / `DP` | Data Parallelism（数据并行）大小——整体复制，不切权重 |
| `N_gpu` | 总 GPU 卡数 |
| `gpusPerNode` | 单节点 GPU 数（典型值 8，如 NVIDIA HGX 基板） |

**精度与字节数**

| 符号 | 含义 |
|---|---|
| `b_w` | 权重每参数字节数，由量化精度决定（FP16=2，FP8=1，FP4=0.5） |
| `b_kv` | KV cache 每元素字节数，由 KV cache 精度决定，与 `b_w` 独立（如权重 INT4 + KV FP8） |
| `b_act` | 激活 / 通信消息精度字节数，通常取 bf16=2，与权重量化精度无关 |

**通信**

| 符号 | 含义 |
|---|---|
| `BW` | 链路带宽（节点内 / 节点间分别取值） |
| `α` | 单次集合调用（per-call）的固定总延迟（见假设 7） |
| `*CommOverlap` | 各并行维度的计算-通信重叠系数（1 = 完全隐藏，0 = 完全暴露） |

**MoE**

| 符号 | 含义 |
|---|---|
| `E` / `experts` | 专家总数 |
| `k` / `expertsPerToken` | 每 token 路由到的专家数（top-k） |
| `coverage` | batch 内专家覆盖率 `1 - (1 - k/E)^B` |

**延迟指标**

| 缩写 | 全称 / 含义 |
|---|---|
| TTFT | Time To First Token，首 token 延迟（= prefill 总耗时） |
| TPOT | Time Per Output Token，每输出 token 耗时 |
| ITL | Inter-Token Latency，token 间延迟（等价于 TPOT） |
| E2E | End-to-End，端到端延迟 |
| MFU | Model FLOPs Utilization，模型算力利用率 |

**架构缩略词**

| 缩写 | 全称 |
|---|---|
| MHA | Multi-Head Attention（多头注意力） |
| GQA | Grouped Query Attention（分组查询注意力） |
| MLA | Multi-head Latent Attention（多头潜在注意力） |
| SSM | State Space Model（状态空间模型） |
| MoE | Mixture of Experts（混合专家模型） |
| PD 分离 | Prefill-Decode Disaggregation（两阶段部署在独立 GPU 池） |

**注意力计算**

| 术语 | 含义 |
|---|---|
| QK^T | Query-Key 点积——计算注意力分数矩阵（`Q × K^T`），FLOPs 为 `2 * N² * qDim` |
| AV | Attention-Value 加权求和——用注意力分数对 Value 加权（`Attn × V`），FLOPs 同为 `2 * N² * qDim` |
| QK^T + AV | 注意力二次项的总和，每层 `4 * N² * qDim` FLOPs |

**派生常量与性能中间量**

| 符号 | 含义 |
|---|---|
| `W_bytes` | 模型权重总字节数 `= P_total × b_w` |
| `flops_per_token` | 每 token 的线性项 FLOPs `≈ 2 × P_active`（不含注意力二次项） |
| `kv_per_token` | 每 token 的 KV cache 字节数（见第四节公式） |
| `Peak_FLOPs` | 单卡峰值算力（FLOPS），取决于量化精度对应的数据类型列 |
| `T_compute` | Prefill 纯计算时间 `= FLOPs_prefill / (N_gpu × Peak_FLOPs × MFU)` |
| `T_comm` | Prefill / Decode 通信时间（TP all-reduce + EP all-to-all + PP P2P 之和） |
| `T_step` | Decode 单步时间（= TPOT = ITL） |
| `T_tp_comm` | Decode 单步中 TP all-reduce 的通信耗时 |
| `S_history` | Decode 阶段当前已生成的历史 token 数（`S = N_in + 已生成数`） |
| `BW_eff` | 单卡有效显存带宽 `= bwGbps × bwEffDecode`（bytes/s） |
| `bwGbps` | GPU 原始 HBM 带宽（GB/s，硬件规格） |
| `bwEffDecode` | Decode 阶段显存带宽利用率校准参数（0~1，IDEAL 模式取 1） |
| `W_read_per_step` | Decode 每步需读取的权重量（dense = `W_bytes`；MoE 取决于 expert coverage） |


## 2. 简化假设

性能估算的目的只是一个粗略的上界参考，因此本项目尽可能简化实际系统中的各种复杂变量，只保留对延迟和吞吐影响最大的因素：

1. **不建模 Continuous Batching 与 Paged Attention**：假设一个 batch 的输入一次性全部送入、一次性计算完成；序列长度 uniform，没有请求的动态进出，也没有 padding 开销。这两项工程优化主要解决显存碎片和动态序列长度问题，在 uniform batch 假设下不影响吞吐结论。
2. **单一量化精度**：整个模型的所有权重量化到同一精度（FP4 / FP8 / FP16 / FP32 之一），用单一的每参数字节数描述。
3. **Workload 执行模型**：workload 采用简化模型，支持两种 prefill / decode 共存模式：
   - **非 PD（Colocated）**：prefill 和 decode 共享同一组 GPU，按 prefill 请求比例 `r`（`prefillRatio`）分时占用。两阶段的资源竞争用概率模型估算：prefill 和 decode 各占 GPU 时间比例 `ρ_prefill` 和 `ρ_decode`，TTFT 和 TPOT 各自叠加对方阶段的期望等待（随机落入对方阶段时，平均等待 = 对方时长 / 2）。
   - **PD 分离（Disaggregated）**：prefill 和 decode 部署在独立的 GPU 池，两阶段不竞争资源。TTFT = prefill 计算时间 + KV cache 跨池传输的暴露部分（`kvTransferOverlap` 控制重叠比例）；TPOT 不受 prefill 影响。

   由此本模型只评估**单请求视角**：理想值模式下 TTFT 仅由 GPU 总数决定（总 FLOPs ÷ 总算力），与布局形态无关；校准模式下 PP 跨节点会通过暴露的通信时间（带宽项 × `(1 - ppCommOverlap)` + α）影响 TTFT，但此模型仍是简化假设——不考虑非均匀输入（batch 内 input/output 长度不一致）带来的 overlap 差异。真实服务系统中 PP + Chunked Prefill 对吞吐与尾部延迟（P99）的收益来自 continuous batching 调度层（长短请求混跑、消除队头阻塞）。多轮对话场景不在本假设范围内。
4. **多节点结果不追求精确**：单节点做严格校准，多节点只加一个低阶校正因子。
5. **流水线 bubble 不建模**：真实调度会用更细的 microbatch 切分、交错执行、chunked prefill 等手段大幅压缩流水线填充 / 排空的空洞，`(B+pp-1)/B` 这类悲观估计会明显高估实际开销。理想值模型假设 bubble ≈ 0，只把阶段间 P2P 激活搬运计入 TTFT（见通信代价通用模型）。
6. **激活显存粗估**：开 FlashAttention 时 activations 近似为单个 `O(N×h)` 残差 buffer；关 FlashAttention 时再叠加 `N²` 注意力分数矩阵（真实的 OOM 悬崖）。不精确建模 buffer 个数与碎片——相对权重 + KV，激活是小项；且与「调度低效」不同，激活是真实物理占用，理想模型保留其下界而非置零。
7. **α（小消息集合通信延迟）的语义与取值**：α 是**单次集合调用**（per-call）的固定总延迟——一次 all-reduce 内部所有轮、所有 hop 已含在内，不是 per-hop；hop 相关的数据量由带宽项 `2(t-1)/t` 负责。理想值模式（IDEAL）取 α=0，输出纯带宽严格上界；实际校准值节点内 ≈ 0.01 ms / 跨节点 ≈ 0.03 ms，与实测 NCCL 小消息 all-reduce 总量 6–11 μs 同量级（不采用 per-hop × hop 数的高估口径）。α 只在对应 `*CommOverlap < 1` 时生效（decode 小 batch 的集合通信无法重叠）。不从 batch=1 推理延迟反推 α（混有小 GEMM kernel 延迟，会高估一个数量级）。


## 3. 建模主脉络：PD两阶段 + 四条资源轴

虽然已经做了大量简化，LLM 推理涉及的变量仍然很多：不同的注意力架构、KV cache 的计算方式、DP/TP/PP/EP/PD 分离等各种并行策略的交叉影响、FlashAttention 的作用、输入输出长度和 prefill/decode 比例的工作负载建模……

怎么把这些因素组织成一个统一的框架？

核心思路其实很简单。我们做的是**静态模型**——假设系统已经处于稳态，只需计算当前 batch 的输入和输出对 KV cache / 吞吐 / 延迟的影响；下一个 batch 会自动刷新 KV cache，每批的显存占用都是一样的。这样就能大幅简化整个建模过程。

具体来说，LLM 推理的数据流可以分为 **prefill 和 decode 两个阶段**，它们的性能瓶颈完全不同：

| 阶段 | 瓶颈轴 | 物理原因 | 时间主公式 |
|---|---|---|---|
| **Prefill**（决定 TTFT） | 算力（compute bound） | 权重只读一次，摊到 `B × N_in` 个 token 上，算术强度高 | `T_prefill ≈ FLOPs / (MFU × Peak_FLOPs)` |
| **Decode**（决定 TPOT / ITL） | 显存带宽（memory bound） | 每生成 1 个 token 都要重读全部权重 + 全部历史 KV cache，算术强度 ≈ 2 FLOP/byte | `T_step ≈ (W_bytes + KV_read_bytes) / BW_eff` |

而系统层面，我们只需要关注**四条资源轴**：

1. **显存容量** —— 决定「能不能跑」（VRAM 检查）
2. **显存带宽** —— 决定 decode 快不快
3. **算力 FLOPs** —— 决定 prefill 快不快
4. **通信带宽** —— 并行引入的第四条轴，影响延迟

把这四条轴和两个阶段组合起来，整个建模的流程就清晰了：

```
(model, quant)      --> 三个派生常量: W_bytes, kv_per_token, flops_per_token
(hardware, layout)  --> 每卡容量 / 带宽 / 通信代价
(B, N_in, N_out)    --> 工作负载
        |
        v
  [1] VRAM 检查 (容量轴) --不满足--> 报告 OOM + 反推 B_max
        | 满足
        v
  [2] Prefill (算力轴 + 通信)  --> TTFT
  [3] Decode    (带宽轴 + 通信)  --> TPOT = ITL
        |
        v
  [4] E2E = TTFT + N_out * TPOT
      throughput = B * N_out / E2E
```

### 从模型规格到三个派生常量

无论什么模型——Dense 或 Mixture of Experts（MoE，混合专家）、标准注意力或 Multi-head Latent Attention（MLA，多头潜在注意力）——第一步都是把模型规格压缩成**三个派生常量**。后续所有的计算都建立在这三个数之上：

```
W_bytes         = P_total * b_w
                  └──┬──┘   └┬┘
                     │       └ 每参数字节数（由量化精度决定）
                     └ 总参数量（MoE 含全部专家）

kv_per_token    → 见第四节 KV cache 通用式

flops_per_token ~= 2 * P_active
                   └┬┘  └───┬───┘
                    │       └ 每 token 激活的参数量（dense = 总参数；MoE = top-k 专家 + 常驻 shared expert）
                    └ 每参数约 2 次浮点运算（一次乘 + 一次加）
```

输入模型规格包括：层数 `L`、隐藏维度 `h`、注意力头配置（`kv_heads`、`head_dim`）、MoE 配置（`experts`、`experts_per_token`、`active_params`）、`max_ctx` 等。

注意力还有一个**二次项**需要单独处理：prefill 阶段每层注意力 `4 * N_in² * (q_heads * head_dim)` FLOPs（QK^T + AV 各 `2 * N² * qDim`，其中 `qDim = q_heads * head_dim`）。这是唯一与序列长度平方相关的计算量。对标准 MHA，`qDim = h`；对 GQA 近似成立；对 MLA（如 DeepSeek V3），qDim 为压缩后的查询投影维度，可能小于 `h`。**此项 FLOPs 与是否开启 FlashAttention 无关**——FlashAttention 只改变显存访问模式（避免写出 `N²` 注意力矩阵），不改变浮点运算次数。

符号约定见 Terminology 节。层索引用下标 `_l` 表示第 `l` 层的取值，`Σ_l` 表示对所有层求和。

这样我们就把「模型规格」和「硬件配置」这两个复杂的输入，压缩成了三个简单的常量（`W_bytes`、`kv_per_token`、`flops_per_token`）加上四条资源轴。后面的一切计算——显存检查、prefill 时间、decode 时间——都只是在这个框架上填数字。

**关键设计原则**：所有场景开关（FlashAttention、量化、Prefill-Decode 分离即 PD 分离、各种并行策略）都不改变这个框架的结构，只改变四条轴上的某个数值。


## 4. KV Cache 建模

KV cache 是连接模型架构和性能计算的桥梁——它同时影响显存容量（能不能装下）和 decode 带宽（每步要读多少数据）。不同的注意力架构，KV cache 的计算方式差异巨大，所以值得单独拿出来建模。

### 通用公式

对标准注意力，KV cache 的通用计算式为：

```
kv_per_token = 2 * b_kv * Σ_l (kv_heads_l * head_dim_l * compress_l)
              └┬┘  └─┬─┘   └───────────┬────────────┘   └────┬───┘
               │     │                 │                     │
               │     │                 │                     └ 该层的 KV 压缩系数（普通层=1）
               │     │                 └ 该层、单个 token、单份（K 或 V）的元素个数
               │     └ 每个元素占几字节（FP16=2, FP8=1）
               └ K 和 V 两份

KV_total(S) = Σ_l 2 * b_kv * kv_heads_l * head_dim_l * min(S, cap_l)
               └┬┘ └┬┘ └─┬─┘ └────────────┬──────────┘   └───┬────┘
                │   │    │                │                  │
                │   │    │                │                  └ 该层实际缓存的 token 数
                │   │    │                │                      （普通层 = S；滑动窗口层封顶在窗口大小）
                │   │    │                └ 该层单份（K 或 V）每 token 的元素个数
                │   │    └ 每个元素占几字节
                │    └ K 和 V 两份
                └ 对所有层求和
```

其中：

- `compress_l`：MLA 层取 `mla_ratio`，其余取 1
- `cap_l`：滑动窗口层取 `sliding_window`，其余取 ∞
- 线性注意力 / State Space Model（SSM，状态空间模型）层：贡献为 0（状态恒定，不随 S 增长）

### 五种 KV 形态

项目模型库中目前支持以下五种 KV 形态：

| KV 形态 | 代表模型 | 数据集字段 | 对 KV_total 的影响 |
|---|---|---|---|
| 标准 MHA / GQA | Llama 3.1 | `kv_heads`、`head_dim` | 基础公式，随 S 线性增长 |
| MLA | DeepSeek V3 | `mla_ratio ≈ 0.0176` | 压缩到基线的约 1/57 |
| 线性注意力层 | Qwen3-Next（GatedDeltaNet） | `linear_attention_layers` | 这些层 KV = 0 |
| SSM 混合层 | Jamba、Nemotron H | `mamba_ratio` | 常数状态，不随 S 增长 |
| 滑动窗口注意力 | Gemma 3（52/62 层，window=1024） | `local_layers` + `sliding_window` | 单层 KV 封顶在 window，长序列增长亚线性 |

一个值得注意的观察：不同模型的 KV cache 增长行为差异巨大。标准 MHA 随序列长度线性增长，MLA 压缩到约 1/57，线性注意力和 SSM 层的贡献为零或常数，滑动窗口则在长序列下呈亚线性增长。这些差异直接决定了「给定显存能装多大的 batch」和「长上下文场景下 decode 会不会被 KV 读取拖慢」。


## 5. VRAM 容量建模

有了 KV cache 的计算方式，显存占用的各个组成部分就都齐了。每卡显存占用：

```
KV_total     = kv_per_token * (N_in + N_out / 2) * B
               └─────┬────┘   └────────┬────────┘   └┬┘
                     │                 │             └ batch size
                     │                 └ 每请求稳态平均序列长度（输入 + 平均一半输出）
                     └ 每 token 的 KV 字节数（见第四节）

# weight sharding: dense 和 MoE 不同
dense:  W_per_gpu = W_bytes / (TP * PP)
MoE:    W_per_gpu = W_nonexpert / (TP * PP) + W_expert / (TP * EP * PP)
                    └─────┬─────┘ └───┬────┘   └──┬──┘   └─────┬─────┘
                          │           │           │            │
                          │           │           │            └ 专家权重的总分片因子（EP 组间分发 × TP 组内切）
                          │           │           └ 全部专家权重
                          │           └ 非专家权重的分片因子（只按 TP、PP 切）
                          └ 非专家权重（attention、shared expert、embedding）

VRAM_per_gpu = W_per_gpu + KV_total / (TP * PP) + Activation / (TP * PP) + overhead
               └───┬────┘   └──────┬─────────┘   └─────────┬──────────┘    └───┬───┘
                   │               │                       │                   └ CUDA 上下文、碎片等预留
                   │               │                       └ 激活值（FlashAttention 开关的落点）
                   │               └ KV cache 属于 attention，按 TP、PP 切
                   └ 每卡权重（见上）
```

- 切分明细与 TP×EP 协作规则见第六节
- **Activation（激活值）是 FlashAttention 开关唯一的落点**：
  - 开 FlashAttention：`O(B * N * h)`，随序列长度线性增长
  - 关 FlashAttention：注意力分数矩阵 `O(B * n_heads * N²)`，长序列出现内存悬崖，直接 Out Of Memory（OOM，显存溢出）

判定与反推：

```
feasible = VRAM_per_gpu <= VRAM_capacity * (1 - headroom)
           └─────┬─────┘  └──────┬──────┘    └────┬────┘
                 │               │                └ 预留余量（碎片、激活峰值等）
                 │               └ 单卡物理显存
                 └ 每卡实际占用（见上）

B_max = VRAM 约束关于 B 的上界（不可行时报告）
```


## 6. 并行建模 = 切分算子 + 通信项

Tensor Parallelism（TP，张量并行）、Pipeline Parallelism（PP，流水线并行）、Expert Parallelism（EP，专家并行）、Data Parallelism（DP，数据并行）——看起来很复杂，但它们其实是同一个模式的四种实例：**切分某类显存占用 + 添加一项通信代价**。

| 并行方式 | 切分什么（显存收益） | 添加什么通信（代价） | 适用场景 |
|---|---|---|---|
| **TP**（size = t） | 权重、KV cache、激活 ÷ t | 每层 2 次 all-reduce，消息量 ∝ `B × N × h` | 仅限节点内（吃高带宽）；prefill / 大 batch 收益大 |
| **PP**（size = p） | 权重、KV cache ÷ p（按层切） | 阶段间点对点传激活（见通信代价通用模型） | 跨节点便宜；流水线 bubble 不建模（见假设 5） |
| **EP**（size = e） | 专家权重 ÷ e（与 TP 的复合见下方协作规则） | 每个 MoE 层 2 次 all-to-all（dispatch + combine） | 仅 MoE 模型 |
| **DP**（size = d） | 不切分（权重全量复制） | 推理期无通信 | 纯吞吐 × d |
| **PD 分离** | 按阶段切分，不按模型切分 | KV cache 从 prefill 池搬到 decode 池，耗时 `KV_total / BW_节点间` | 让两个池各自选最优并行与硬件配置 |

约束方程：

```
 TP * EP * PP * DP = N_gpu
└┬┘  └┬┘  └┬┘  └┬┘   └─┬─┘
 │    │    │    │      └ 总卡数
 │    │    │    └ 数据并行：整体复制，不切权重
 │    │    └ 流水线并行：按层切
 │    └ 专家并行：分发专家（仅 MoE；dense 强制 = 1）
 └ 张量并行：层内切权重（通信吃带宽，必须落在同一节点内）
```

### TP×EP 协作规则（仅 MoE）

TP 与 EP 是两条独立的轴、占不同的卡，总卡数 = TP × EP × PP × DP。关键不变量：**专家权重无论怎么切，都摊到全部卡上**——`W_expert / (TP×EP) = W_expert / N_gpu`。所以 TP/EP 的切法不改变专家显存，只改变「非专家权重」和「通信方式」。

8 卡三种切法对照（PP=DP=1）：

| 布局 | 专家权重/卡 | 非专家权重/卡（attention 等） | 通信 |
|---|---|---|---|
| TP=8, EP=1 | W_expert/8 | W_nonexpert/8 | all-reduce 跨 8 卡（贵，需 NVLink） |
| TP=1, EP=8 | W_expert/8 | W_nonexpert/1（全复制） | all-to-all 跨 8 卡 |
| TP=2, EP=4 | W_expert/8 | W_nonexpert/2 | 组内 all-reduce（2 卡）+ 组间 all-to-all（4 组） |

取舍：

- **拉高 TP**：非专家权重切得碎、省显存，但 all-reduce 横跨所有卡，吃带宽、仅限节点内
- **拉高 EP**：all-to-all 通信更省，但非专家权重在各组复制、吃显存
- **混合（TP=2, EP=4）**：两者折中

#### EP 与延迟的关系（易误解）

固定卡数下，每卡专家字节只取决于乘积 `TP×EP`，EP 切多少并不比 TP 更能降每卡字节；且非专家权重只有 TP 切、EP 复制，所以同卡数下 TP 反而 TPOT 更低。EP 的真正价值是**规模扩展**：TP 被单节点卡死（`TP ≤ 节点内卡数`），模型大到一个节点摊不薄每卡字节时，只能靠 EP 把专家切到别的节点、继续用更多卡摊薄每卡读取、压低 TPOT。即「EP 管扩展、不管每卡效率」，代价是跨节点 all-to-all 更慢；且加卡降 TPOT 收益递减，最终撞通信墙。


### 通信原语对照：all-reduce vs all-to-all

一句话：all-reduce 是**聚合**（每人出一份，大家拿到总和；有计算、数据位置不变）；all-to-all 是**重分发**（数据按目的地搬到别的卡；无计算、数据位置改变）。

```
all-reduce（求和）：
前: GPU0:[a0]  GPU1:[a1]  GPU2:[a2]  GPU3:[a3]     ← 各持一份同形状张量
后: GPU0:[Σa]  GPU1:[Σa]  GPU2:[Σa]  GPU3:[Σa]     ← 每卡都拿到总和

all-to-all（按目的地搬运）：
前: GPU0:[x00 x01 x02 x03]                          ← x0j = GPU0 上要发给 GPUj 的块
    GPU1:[x10 x11 x12 x13]  ...
后: GPU0:[x00 x10 x20 x30]                          ← GPU0 收齐所有「发给我的」块
    GPU1:[x01 x11 x21 x31]  ...
```

| | all-reduce | all-to-all |
|---|---|---|
| 语义 | 求和聚合：每卡拿到总和 | 个性化交换：数据到达目的卡 |
| 用在哪 | **TP**：切权重后每卡算出部分和，需求和 | **EP**：token 要搬到持有其选中专家的卡 |
| 出现次数 | 每层 2 次（attention / MLP 输出后） | 每个 MoE 层 2 次（dispatch 去 + combine 回） |
| 每卡流量（t 卡） | 收/发各 `2(t-1)/t × msg`（ring通信） | 收/发各 `(t-1)/t × msg` |
| 通信模式 | 固定，与数据内容无关 | 依赖路由结果（每 token 选了哪个专家），可能不均衡 |
| 对互连的要求 | 规则、易优化（ring / tree） | 全对全交换，跨节点扩展性更差 |

**哪个成本更高？** 分两层看，结论不同。

单次调用、同样 `msg` → all-reduce 贵约 2 倍（ring all-reduce = reduce-scatter + allgather 两遍）。但真实负载里消息量不同：

```
TP: 每层 2 次 AR × 每次 ~2(t-1)/t × (B*N*h*b_act)
    → 每卡每层 ≈ 4 × B*N*h*b_act

EP: 每 token 复制给 k 个专家，但每卡只处理 1/t 的 token，dispatch + combine 合计
    → 每卡每 MoE 层 ≈ 2 × B*N*k*h*b_act / t

EP / TP ≈ k / (2t)        # k = experts_per_token
```

典型 `k=8, t=8` → 比值 ≈ 0.5：EP 每层通信量反而只有 TP 的一半，且只出现在 MoE 层（TP 通信在每一层都有）、可与专家计算重叠——这是大型 MoE 用 EP + DP 而非巨型 TP 组跨节点的通信侧原因。

all-to-all 省下的字节数是用两个隐性成本换的：① 路由不均衡（热门专家的卡多收多算，长尾等待）；② 拓扑依赖（跨节点 InfiniBand 上全对全交换扩展性差）。

> 本文不考虑 MoE 路由不均衡的场景，假设专家被均匀路由。

### 通信代价的通用模型（带宽项 + 延迟项，缺一不可）

通信时间统一用 LogP 形式：`T = msg / BW + α × hops`。只建带宽项会在 decode 小消息场景系统性失真。

- **TP（ring all-reduce）**：

```
T_ar = n_ar * ( 2(t-1)/t * msg / BW + α )
        └┬┘     └──┬───┘   └┬┘  └┬┘  └┬┘
         │         │        │    │    └ 单次集合调用的固定总延迟（含所有轮和 hop）
         │         │        │    └ 链路带宽
         │         │        └ 单次消息量（prefill: B*N*h*b_act；decode: B*h*b_act）
         │         └ ring 单次 all-reduce 的总传输量（reduce-scatter + all-gather）
         └ 每层 all-reduce 次数 = 2（attention 输出后 + MLP 输出后）
```
- **EP（all-to-all）**：每个 MoE 层 dispatch + combine 两次，带宽项与 α 项同构
- **PP**：阶段间点对点传激活。每条阶段间链路传输整个 batch 的隐藏态 `B × N_in × h × b_act`；`(pp-1)` 条链路并行，故 makespan = `B × N_in × h × b_act / BW`。带宽按「单节点→节点内、多节点→跨节点」启发式选；流水线填充/排空（bubble）不建模（见假设 5），P2P 延迟为低阶项略去
- **PD 分离**：KV cache 一次性搬运，`KV_total / BW_节点间`

两个物理细节值得展开：

**1. 每跳延迟 α 必须单独建模。** 通信时间 = 带宽项 + `α × hops`，带宽项只算搬字节，每一步的握手（doorbell 事务、DMA 启动、交换机转发、完成同步）是与消息大小无关的固定成本。小消息世界由跳数主导——最直接的证据是 NCCL 按消息大小切算法：大消息走 ring（带宽最优，延迟 `2(t-1)` 步），小消息走 tree（延迟 `2×log₂(t)` 步：自底向上归约 `log₂(t)` + 自顶向下广播 `log₂(t)`）。t=64 时 126 步 vs 12 步，小消息延迟差一个量级。两个精化：NCCL 实际用 double binary tree（两棵互补树各搬一半数据，带宽接近 ring、延迟保持 log）；NVSwitch 硬件上 NVLS（NVLink SHARP）在交换机内归约，进一步塌缩到约 1 跳。

- **建模含义**：小消息有效跳数由算法 / 硬件决定（ring / tree / NVLS 自动切换），第一性原理推不准；本文校准阶段直接用 `nccl-tests` 测目标机器小消息（8 字节）all-reduce **总延迟**，作为整体常数 `α_collective(t, 拓扑)`，不逐跳分解。
- decode 阶段 TP 消息极小（`B × h`），通信时间由 α 主导而非带宽——这正是「TP 在小 batch 下反而亏」的定量来源。分界点 `msg = α_total × BW`（节点内约数 MB 量级）：之上带宽主导，之下延迟主导。
- **α 数值建模**：实测是节点内 8 卡小消息 all-reduce **总延迟** ~6–11μs（NCCL 2.27 报 ~6.3μs；LL/LL128 协议每步 ~1–2μs），IB NDR 点对点硬件延迟 <1μs。α 属校准参数：用目标机器 nccl-tests 实测小消息延迟定标；理想值模式取 α=0，输出纯带宽严格上界。

**2. 消息精度用激活精度 `b_act`**（通常 bf16），与权重量化精度 `b_w` 无关：权重 INT4 不会让 all-reduce 消息变小。


## 7. 延迟建模

有了前面的基础设施（派生常量、KV cache 模型、显存检查、并行通信模型），延迟计算就是把各个部分填进去。

**TTFT（prefill，算力轴）：**

```
FLOPs_prefill = B * N_in * flops_per_token + 4 * L * N_in^2 * (q_heads * head_dim) * B
                └────────┬───────────────┘   └─────────────────┬─────────────────────┘
                         │                                     └ 注意力二次项（QK^T + AV）：唯一随序列长度平方
                         │                                        增长的项（与 FlashAttention 无关，见第三节）
                         └ 矩阵乘项：总 token 数 × 每 token FLOPs

T_compute = FLOPs_prefill / (N_gpu * Peak_FLOPs * MFU)
                            └────────────┬────────────┘
                                         └ 集群有效算力（卡数 × 单卡峰值 × 利用率）

T_comm = TP all-reduce per layer + EP all-to-all + PP P2P
TTFT   = T_compute + non_overlapped_comm (+ KV transfer if PD-disaggregated)
```

**TPOT（decode，带宽轴）：**

```
bytes_per_step = W_read_per_step / (t * p) + kv_per_token / (t * p) * S_history * B
                 └───────┬───────┘ └──┬──┘   └──────────────┬─────────────────────┘
                         │            │                     └ 每卡要读的历史 KV 总量
                         │            │                       （每 token KV ÷ 分片 × 历史长度 × batch）
                         │            └ 分片因子（权重按 TP × PP 切）
                         └ 每步要读的权重量（dense = 全量；MoE 随 batch 专家覆盖率变化，见下）

T_step = bytes_per_step / BW_eff + T_tp_comm
         └───────┬──────┘ └─┬─┘    └───┬───┘
                 │          │          └ TP 通信耗时（见第六节通用式）
                 │          └ 单卡有效显存带宽（= gpu.bwGbps × bwEffDecode）
                 └ 每卡每步读取的总字节数

TPOT = ITL = T_step
```

**MoE 注意**：第一项不是全量权重。非专家权重（attention、shared expert、embedding）每步必读；专家权重的读取量取决于 batch 内的专家覆盖率 `1 - (1 - k/E)^B`——B 小时只读 top-k 命中的专家，B 增大时趋近全量专家集合。这个覆盖率是**每步读取量的硬下界，流水线降不掉**：本步必须给全部 B 个 token 算完 FFN 才能进入下一步，被命中的专家至少要读一次；重叠只能藏延迟、减不了字节，而 decode 恰恰被字节（带宽）卡住。唯一能低于它的是跨 step 专家缓存（依赖路由局部性），本文不建模、按每步全量重读。dense 模型无此问题，`W_read_per_step = W_bytes` 恒成立。

**TP 注意**：decode 阶段 TP 的 all-reduce 消息很小（`B × h`），此时通信是延迟受限而非带宽受限——这正是「TP 在小 batch 下反而亏」的来源。而 TP 对 decode 的收益在于聚合多卡 HBM 带宽，两头对冲后是否划算由模型自动算出。

### Speculative Decoding（投机解码）

Speculative Decoding 用一个小而快的 draft 模型先自回归生成 γ 个候选 token，再由主模型一次性并行验证。验证通过的 token 直接接受，不通过则从分歧点重新 draft。

```
expected_tokens_per_cycle = γ * acceptance_rate + 1
                            └┬┘  └────┬─────┘    └┬┘
                             │        │            └ 验证步本身也产出 1 个确定 token
                             │        └ draft token 的接受率（取决于 draft 模型质量）
                             └ draft 步数（每步 1 个 token）

cycle_time = γ * draft_step_ms + verify_step_ms
             └┬┘  └─────┬────┘    └─────┬────┘
              │         │               └ 主模型验证步（1 次前向，等价于 1 步 decode）
              │         └ draft 模型每步耗时（通常远小于主模型）
              └ 提议长度

TPOT_sd = cycle_time / expected_tokens_per_cycle
```

关键取舍：γ 越大，每周期产出的期望 token 数越多，但接受率随 γ 增大而下降（draft 偏离主模型的概率累积），且验证步的显存和计算开销不变。最优 γ 取决于 draft 模型与主模型的分布接近程度。两个模型同时驻留显存，VRAM 检查需额外计入 draft 模型权重。


## 8. 指标汇总：吞吐和端到端延迟

最后一步是把 prefill 和 decode 的时间串起来，得到系统级指标：

**PD 分离模式**（两阶段独占 GPU 池，无竞争）：

```
E2E_latency = TTFT + N_out * TPOT
              └┬┘    └────┬─────┘
               │          └ decode 总耗时（每步耗时 × 输出长度）
               └ 首 token 延迟（= prefill 计算 + KV 传输暴露部分）
```

**非 PD 模式**（两阶段共享 GPU，概率竞争）：

```
TTFT_eff = TTFT + ρ_decode * TPOT * N_out / 2
           └┬┘    └───────────┬──────────────┘
            │                 └ prefill 随机落入 decode 阶段时的期望等待
            │                        （ρ_decode = decode 占用 GPU 时间比例）
            └ 纯 prefill 计算时间

TPOT_eff = TPOT + ρ_prefill * TTFT / 2
           └┬┘    └────────┬──────────┘
            │              └ decode 步随机落入 prefill 阶段时的期望等待
            │                      （ρ_prefill = prefill 占用 GPU 时间比例）
            └ 纯 decode 单步时间

E2E_latency = TTFT_eff + N_out * TPOT_eff
```

其中 `ρ_prefill` 和 `ρ_decode` 由 `prefillRatio`（`r`）和各阶段的吞吐率推算得出（见第二节假设 3）。

**吞吐（两种模式通用）**：

```
throughput = B * N_out / E2E_latency      # output tokens per second
             └────┬───┘   └────┬────┘
                  │            └ 总耗时（代入对应模式的 E2E）
                  └ 总产出 token 数（宏观吞吐 = 总产出 ÷ 总耗时）
```

微观和宏观两个视角必须区分，否则建模时容易概念混淆：

- **微观视角（Per-Request Latency，单请求延迟）**：一个请求从进来到离开卡了多久。核心指标是 TTFT 和 TPOT，对应用户体验与服务等级协议（Service Level Agreement，SLA）。
- **宏观视角（System Throughput，系统吞吐）**：这组 GPU 每秒产出多少有效 token。核心指标是 output tokens/sec，对应硬件利用率和成本效益。

### 利用率指标

除了延迟和吞吐，计算器还输出两个利用率指标，帮助判断系统瓶颈在哪条轴上：

- **Prefill 算力利用率**：实际 FLOPS/s ÷ 峰值 FLOPS/s。低于 MFU 校准值说明通信或调度在拖后腿；接近 1 说明算力是瓶颈。
- **Decode 带宽利用率**：实际 bytes/s ÷ 峰值 bytes/s。低于 `bwEffDecode` 校准值说明通信暴露或 KV cache 读取不充分；接近 1 说明带宽是瓶颈。

### 一个关键洞察：开关只改常数，不改建模结构

回过头来看，整个框架的设计有一个很重要的性质——所有的「场景开关」都不改变计算流程，只改变四条轴上的某些数值：

| 开关 | 影响哪条轴 | 模型内的体现 |
|---|---|---|
| FlashAttention 开 / 关 | 容量轴 | 关：激活出现 `O(B * n_heads * N²)` 项，长序列显存悬崖（易 OOM）；FLOPs 不受影响 |
| 量化精度 | 容量轴 + 两阶段时间 | 选择精度同时决定两个量：`b_w`（缩放 `W_bytes`）和 GPU 数据中对应精度的 `Peak_FLOPs` 列（bf16 / fp8 / int4 / fp4）；KV cache 量化由独立的 `b_kv` 控制 |
| PD 分离 | 通信轴 | TTFT 加 KV cache 传输项；prefill / decode 池使用各自独立的并行布局 |
| MoE | 派生常量 | `W_bytes` 含全部专家，`flops_per_token` 用激活参数；EP 变为可选 |
| Speculative Decoding | 带宽轴（decode） | TPOT 替换为 `(γ × draft_step + verify_step) / (γ × α + 1)`；VRAM 额外计入 draft 模型权重 |
| PD 非 PD 共存模式 | 延迟轴 | TTFT / TPOT 叠加对方阶段的期望等待（见第二节假设 3） |

这意味着：增加一个新的场景开关（比如未来支持某种新的注意力变体），只需要确定它影响哪条轴、改变哪个数值，整个计算流程保持不变。


## 9. 校准方法论

上述建模计算采用的是理论峰值上界（算力打满、带宽打满、通信完美重叠）。纯峰值是严格上界，与实测通常有数倍差距，因此预留以下**物理层**效率常数供用户调节：

- `mfuPrefill`：prefill 阶段实际算力利用率
- `bwEffDecode`：decode 阶段有效显存带宽利用率
- `commEffIntra` / `commEffInter`：节点内 / 节点间通信带宽效率
- `tpCommOverlap` / `epCommOverlap` / `ppCommOverlap`：各并行方式的计算-通信重叠系数（1 = 完全隐藏，0 = 完全暴露）
- `alphaIntraMs` / `alphaInterMs`：小消息集合通信固定延迟（理想值模式取 0）

校准只引入硬件 / 物理层面的修正，**未引入 vLLM / SGLang 等框架的引擎级校准因子**——它们的稳态吞吐来自 continuous batching 前提，与本模型的单 batch 假设不可比（见附录）。单节点用实测数据拟合这几个常数即可，参数空间小，校准是良态问题。多节点不追求精确，只额外引入一个关于通信量 / 跳数的校正因子，形式限于线性、最多二次模型（与第二节简化假设 4 对应）。

### 校准参数拟合

我们将公开实测数据（NVIDIA 系列，含来源链接与协议分类）收集在 [`src/data/calibration`](../src/data/calibration)，基于本模型分别计算了理想值和校准值，并与实测结果进行了对照，详细数据见 [calibration README](../calibration/README.md)。

基于上述对照，本文拟合出如下一组校准预设作为计算器的默认值（可在 UI 中手动调节，也支持重置到理想值）：

```
mfuPrefill:            0.6       # 锚点：~0.17 (llama.cpp) / ~0.5 (TRT-LLM)，取偏保守值
bwEffDecode:           0.55      # 锚点：~0.53
commEff(Intra/Inter):  0.9       # NCCL 大消息典型效率，尚无 nccl-tests 实测锚点
(TP/Ep/PP)CommOverlap: 0.5       # 保守起步，实测可上调
alphaIntraMs:          0.01      # ~10μs（NVLink 小消息 all-reduce 总延迟）
alphaInterMs:          0.03      # ~30μs（InfiniBand 小消息 all-reduce 总延迟）
```

## 10. 计算器演示

理论讲完了，来实际上手。

![demo](../assets/demo.gif)

- → **在线体验**：[llm-inference-calculator](https://llm-inference-calculator-delta.vercel.app/)
- → **源码**：[GitHub](https://github.com/hny0305lin/llm-inference-calculator)（欢迎 Star ⭐ 和贡献 PR）


## 附录：建模问题归档

**问题 1：prefill 和 decode 同时存在时，GPU 吞吐怎么算？取最慢的？流水线？还是并发？**

结论：取决于执行模式。PD 分离模式下两阶段严格串行于各自的 GPU 池，`E2E = TTFT + N_out × TPOT`。非 PD（Colocated）模式下两阶段共享 GPU，用概率竞争模型估算各自的期望等待（见第二节假设 3），公式形式不变但 TTFT 和 TPOT 已包含争用开销。真正的「混跑并发」需要 continuous batching 系统（如 SGLang / vLLM 的运行时调度），不在本项目范围内。

**问题 2：SGLang 的 benchmark 吞吐是直接算出来的吗？**

结论：SGLang 报告的吞吐来自 continuous batching 下大量请求混跑、持续填满硬件的稳态测量，与本项目的单 batch 串行模型前提不同，数字不可直接对比。
