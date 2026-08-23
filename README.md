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

## 三、Step 0：数据输入与派生常量

### 3.1 数据来源

- **模型库**：字段 schema 参考 tps 项目的 `src/data/models`（403 个模型）。**只参考字段结构**，用于检查建模遗漏；其数值与计算逻辑不照搬（已有计算被验证不准），vLLM / SGLang 等引擎级校准因子也不引入——本项目只计算理想值。
- **GPU 库**：字段 schema 参考同项目 `src/data/gpus`（数值需重新校验后再用）：

| GPU 字段 | 对应资源轴 |
|---|---|
| `vram`（GB） | 容量轴 |
| `bw`（GB/s） | 带宽轴（decode） |
| `bf16` / `fp8` / `int8` / `int4` / `fp4`（TFLOPS） | 算力轴，按量化精度选列 |
| `nvlink_bw`（GB/s） | 通信轴（节点内）；为 null 即 PCIe 互连 |

节点间带宽（InfiniBand 等）不入库，由用户输入。

范围备注：tps **没有** PD 分离的估算（其 TP/PP/EP/DP 四种并行都有），PD 分离的 KV 传输项与双池布局是本项目自行定义的部分，无现成参考。

### 3.2 模型字段 → 三个派生常量

输入模型规格：层数 `L`、隐藏维度 `h`、注意力头配置（`kv_heads`、`head_dim`）、Mixture of Experts（MoE，混合专家）配置（`experts`、`experts_per_token`、`active_params`、`moe_execution`）、`max_ctx`。

只派生三个常量，后续全部复用：

```
W_bytes         = P_total * b_w
                  └──┬──┘   └┬┘
                     │        └ 每参数字节数（由量化精度决定）
                     └ 总参数量（MoE 含全部专家）

kv_per_token    → 见 3.3 通用式

flops_per_token ~= 2 * P_active
                   └┬┘  └───┬───┘
                    │         └ 每 token 激活的参数量（dense = 总参数；MoE = top-k 专家 + 常驻 shared expert）
                    └ 每参数约 2 次浮点运算（一次乘 + 一次加）
```

Attention（注意力）的二次项单独记：prefill 阶段每层注意力约 `4 * N^2 * (q_heads * head_dim)` FLOPs，这是唯一与序列长度平方相关的计算量。

**记号约定**（全文通用）：

| 符号 | 含义 |
|---|---|
| `b_w` | 权重每参数字节数，由量化精度决定（FP16=2，FP8=1，FP4=0.5） |
| `b_kv` | KV cache 每元素字节数，由 KV cache 精度决定，与 `b_w` 独立（如权重 INT4 + KV FP8） |
| `b_act` | 激活 / 通信消息精度字节数，通常取 bf16=2，与权重量化精度无关 |
| `h` / `L` | 隐藏维度（残差流宽度）/ 层数 |
| 下标 `_l` | 第 `l` 层的取值（层索引）；`Σ_l` = 对所有层求和 |
| `N_in` / `N_out` | 输入 / 输出序列长度；`B` = batch size；`S` = 序列长度（含历史） |

### 3.3 KV cache 通用式（基础公式只对标准注意力成立）

```
kv_per_token = 2 * b_kv * Σ_l (kv_heads_l * head_dim_l * compress_l)
              └┬┘  └─┬─┘   └───────────┬────────────┘   └───┬───┘
               │      │                 │                     │
               │      │                 │                     └ 该层的 KV 压缩系数（普通层=1）
               │      │                 └ 该层、单个 token、单份（K或V）的元素个数
               │      └ 每个元素占几字节（FP16=2, FP8=1）
               └ K 和 V 两份

KV_total(S) = Σ_l 2 * b_kv * kv_heads_l * head_dim_l * min(S, cap_l)
               └┬┘ └┬┘ └─┬─┘ └────────────┬──────────┘   └────┬────┘
                │    │    │                │                    │
                │    │    │                │                    └ 该层实际缓存的 token 数
                │    │    │                │                      （普通层 = S；滑动窗口层封顶在窗口大小）
                │    │    │                └ 该层单份（K 或 V）每 token 的元素个数
                │    │    └ 每个元素占几字节
                │    └ K 和 V 两份
                └ 对所有层求和
```

- `compress_l`：Multi-head Latent Attention（MLA，多头潜在注意力）层取 `mla_ratio`，其余取 1
- `cap_l`：滑动窗口层取 `sliding_window`，其余取 ∞
- 线性注意力 / State Space Model（SSM，状态空间模型）层：贡献为 0（状态恒定，不随 S 增长）

模型库中存在的五种 KV 形态及对应字段：

| KV 形态 | 代表模型 | 数据集字段 | 对 KV_total 的影响 |
|---|---|---|---|
| 标准 MHA / GQA | Llama 3.1 | `kv_heads`、`head_dim` | 基础公式，随 S 线性 |
| MLA | DeepSeek V3 | `mla_ratio ≈ 0.0176` | 压缩到基线的约 1/57 |
| 线性注意力层 | Qwen3-Next（GatedDeltaNet） | `linear_attention_layers` | 这些层 KV = 0 |
| SSM 混合层 | Jamba、Nemotron H | `mamba_ratio` | 这些层为常数状态，不随 S 增长 |
| 滑动窗口注意力 | Gemma 3（52/62 层，window=1024） | `local_layers` + `sliding_window` | 单层 KV 封顶在 window，长序列增长亚线性 |

### 3.4 已知数据缺口

- **查询头数 `q_heads` 缺失**：数据集只有 `kv_heads`。对 `q_heads × head_dim ≠ hidden_size` 的架构（如 Qwen3-Next、Gemma），注意力二次项无法精确计算；第一版用 `hidden_size` 近似，后续考虑补录 `q_heads` 字段。

## 四、Step 1：VRAM 模型（容量轴）

每卡显存占用：

```
KV_total     = kv_per_token * (N_in + N_out) * B
               └─────┬────┘   └──────┬──────┘   └┬┘
                     │                 │           └ batch size
                     │                 └ 每请求总序列长度（输入 + 输出）
                     └ 每 token 的 KV 字节数（见 3.3）

# weight sharding: dense and MoE differ
dense:  W_per_gpu = W_bytes / (TP * PP)
MoE:    W_per_gpu = W_nonexpert / (TP * PP) + W_expert / (TP * EP * PP)
                    └─────┬─────┘   └───┬────┘   └──┬──┘   └─────┬─────┘
                          │              │           │             │
                          │              │           │             └ 专家权重的总分片因子（EP 组间分发 × TP 组内切）
                          │              │           └ 全部专家权重
                          │              └ 非专家权重的分片因子（只按 TP、PP 切）
                          └ 非专家权重（attention、shared expert、embedding）

VRAM_per_gpu = W_per_gpu + KV_total / (TP * PP) + Activation / (TP * PP) + overhead
               └───┬────┘   └───┬────┘           └────┬────┘           └───┬───┘
                   │              │                     │                    └ CUDA 上下文、碎片等预留
                   │              │                     └ 激活值（FlashAttention 开关的落点）
                   │              └ KV cache 属于 attention，按 TP、PP 切
                   └ 每卡权重（见上）
```

- 切分明细与 TP×EP 协作规则见第五节
- **Activation（激活值）是 FlashAttention 开关唯一的落点**：
  - 开 FlashAttention：`O(B * N * h)`，随序列长度线性增长
  - 关 FlashAttention：注意力分数矩阵 `O(B * n_heads * N^2)`，长序列出现内存悬崖，直接 Out Of Memory（OOM，显存溢出）

判定与反推：

```
feasible = VRAM_per_gpu <= VRAM_capacity * (1 - headroom)
           └─────┬─────┘  └──────┬──────┘    └────┬────┘
                 │                 │                └ 预留余量（碎片、激活峰值等）
                 │                 └ 单卡物理显存
                 └ 每卡实际占用（见上）

B_max = VRAM 约束关于 B 的上界（不可行时报告）
```

## 五、Step 2：并行 = 切分算子 + 通信项

Tensor Parallelism（TP，张量并行）、Pipeline Parallelism（PP，流水线并行）、Expert Parallelism（EP，专家并行）、Data Parallelism（DP，数据并行）不各建一套模型，它们是同一个模式的不同实例：**切分某类显存占用 + 添加一项通信代价**。

| 并行方式 | 切分什么（显存收益） | 添加什么通信（代价） | 适用场景 |
|---|---|---|---|
| **TP**（size = t） | 权重、KV cache、激活 ÷ t | 每层 2 次 all-reduce，消息量 ∝ `B × N × h` | 仅限节点内（吃高带宽）；prefill / 大 batch 收益大 |
| **PP**（size = p） | 权重、KV cache ÷ p（按层切） | 阶段间点对点传激活，量小；但有 bubble（空洞）`(p-1)/m`（m = microbatch 数） | 跨节点便宜；m 小时 bubble 伤延迟 |
| **EP**（size = e） | 专家权重 ÷ e（与 TP 的复合见下方协作规则） | 每个 MoE 层 2 次 all-to-all（dispatch + combine，分发 + 合并） | 仅 MoE 模型 |
| **DP**（size = d） | 不切分（权重全量复制） | 推理期无通信 | 纯吞吐 × d |
| **PD 分离** | 按阶段切分，不按模型切分 | KV cache 从 prefill 池搬到 decode 池，耗时 `KV_total / BW_节点间` | 让两个池各自选最优并行与硬件配置 |

约束方程：

```
TP * EP * PP * DP = N_gpu
└┬┘  └┬┘  └┬┘  └┬┘   └─┬─┘
 │     │    │    │      └ 总卡数
 │     │    │    └ 数据并行：整体复制，不切权重
 │     │    └ 流水线并行：按层切
 │     └ 专家并行：分发专家（仅 MoE；dense 强制 = 1）
 └ 张量并行：层内切权重（通信吃带宽，必须落在同一节点内）
```

**TP×EP 协作规则（仅 MoE）**：TP 与 EP 是两条独立的轴、占不同的卡，总卡数 = TP × EP × PP × DP。关键不变量：**专家权重无论怎么切，都摊到全部卡上**——`W_expert / (TP×EP) = W_expert / N_gpu`。所以 TP/EP 的切法不改变专家显存，只改变「非专家权重」和「通信方式」。

8 卡三种切法对照（PP=DP=1）：

| 布局 | 专家权重/卡 | 非专家权重/卡（attention 等） | 通信 |
|---|---|---|---|
| TP=8, EP=1 | W_expert/8 | W_nonexpert/8 | all-reduce 跨 8 卡（贵，需 NVLink） |
| TP=1, EP=8 | W_expert/8 | W_nonexpert/1（全复制） | all-to-all 跨 8 卡 |
| TP=2, EP=4 | W_expert/8 | W_nonexpert/2 | 组内 all-reduce(2 卡) + 组间 all-to-all(4 组) |

取舍：

- **拉高 TP**：非专家权重切得碎、省显存，但 all-reduce 横跨所有卡，吃带宽、仅限节点内
- **拉高 EP**：all-to-all 通信更省，但非专家权重在各组复制、吃显存
- **混合（TP=2, EP=4）**：两者折中

典型错误：指定 EP=8 后仍按默认 TP=N_gpu 计算，乘积超出卡数（声称 64 卡实际 8 卡），等于把专家权重除了两遍——显存被低估 EP 倍、通信被重复计。即「不要 TP=8 再叠 EP=8」这条守卫。

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
| 每卡流量（t 卡） | 收/发各 `2(t-1)/t × msg`（ring） | 收/发各 `(t-1)/t × msg` |
| 通信模式 | 固定，与数据内容无关 | 依赖路由结果（每 token 选了哪个专家），可能不均衡 |
| 对互连的要求 | 规则、易优化（ring / tree） | 全对全交换，跨节点扩展性更差 |

**哪个成本更高**：分两层，结论不同。

单次调用、同样 `msg` → all-reduce 贵约 2 倍（ring all-reduce = reduce-scatter + allgather 两遍）。但真实负载里消息量不同：

```
TP: 每层 2 次 AR × 每次 ~2(t-1)/t × (B*N*h*b_act)
    → 每卡每层 ≈ 4 × B*N*h*b_act

EP: 每 token 复制给 k 个专家，但每卡只处理 1/t 的 token，dispatch + combine 合计
    → 每卡每 MoE 层 ≈ 2 × B*N*k*h*b_act / t

EP / TP ≈ k / (2t)        # k = experts_per_token
```

典型 `k=8, t=8` → 比值 ≈ 0.5：EP 每层通信量反而只有 TP 的一半，且只出现在 MoE 层（TP 通信在每一层）、可与专家计算重叠——这是大型 MoE 用 EP + DP 而非巨型 TP 组跨节点的通信侧原因。

all-to-all 省下的字节数是用两个隐性成本换的：① 路由不均衡（热门专家的卡多收多算，长尾等待）；② 拓扑依赖（跨节点 InfiniBand 上全对全交换扩展性差）。

### 通信代价的通用模型（带宽项 + 延迟项，缺一不可）

通信时间统一用 LogP 形式：`T = msg / BW + α × hops`。只建带宽项会在 decode 小消息场景系统性失真。

- **TP（ring all-reduce）**：

```
T_ar = n_ar * ( 2(t-1)/t * msg / BW + α * 2(t-1) )
        └┬┘     └────┬────┘   └┬┘  └┬┘  └┬┘  └──┬──┘
         │            │         │    │   │       └ 单次 all-reduce 经过的跳数
         │            │         │    │   └ 每跳启动延迟（节点内 ~8μs / 跨节点 ~25μs）
         │            │         │    └ 链路带宽
         │            │         └ 单次消息量（prefill: B*N*h*b_act；decode: B*h*b_act）
         │            └ ring 单次 all-reduce 的总传输量（reduce-scatter + all-gather）
         └ 每层 all-reduce 次数 = 2（attention 输出后 + MLP 输出后）
```
- **EP（all-to-all）**：每个 MoE 层 dispatch + combine 两次，带宽项与 α 项同构
- **PP**：阶段间点对点传激活，消息量小，通常可忽略
- **PD 分离**：KV cache 一次性搬运，`KV_total / BW_节点间`

两个物理细节：

1. **每跳延迟 α 必须单独建模**：通信时间 = 带宽项 + `α × hops`，带宽项只算搬字节，每一步的握手（doorbell 事务、DMA 启动、交换机转发、完成同步）是与消息大小无关的固定成本；小消息世界由跳数主导的最直接证据是 NCCL 按消息大小切算法——大消息走 ring（带宽最优，延迟 `2(t-1)` 步），小消息走 tree（延迟 `2×log₂(t)` 步：自底向上归约 `log₂(t)` + 自顶向下广播 `log₂(t)`）。t=64 时 126 步 vs 12 步，小消息延迟差一个量级。两个精化：NCCL 实际用 double binary tree（两棵互补树各搬一半数据，带宽接近 ring、延迟保持 log）；NVSwitch 硬件上 NVLS（NVLink SHARP）在交换机内归约，进一步塌缩到约 1 跳。
   - **建模含义**：小消息有效跳数由算法 / 硬件决定（ring / tree / NVLS 自动切换），第一性原理推不准；校准阶段直接用 `nccl-tests` 测目标机器小消息（8 字节）all-reduce **总延迟**，作为整体常数 `α_collective(t, 拓扑)`，不逐跳分解。
   - decode 阶段 TP 消息极小（`B × h`），通信时间由 α 主导而非带宽——这正是「TP 在小 batch 下反而亏」的定量来源。分界点 `msg* = α_total × BW`（节点内约数 MB 量级）：之上带宽主导，之下延迟主导。
   - **α 数值警示**：此前写的「节点内 ~8μs/hop、跨节点 ~25μs/hop」来自 tps 项目的工程估计，与公开实测明显不符——实测是节点内 8 卡小消息 all-reduce **总延迟** ~6–11μs（NCCL 2.27 报 ~6.3μs；LL/LL128 协议每步 ~1–2μs），IB NDR 点对点硬件延迟 <1μs。α 属校准参数：用目标机器 nccl-tests 实测小消息延迟定标；理想值模式（v1）可取 α=0，输出纯带宽严格上界。
2. **消息精度用激活精度 `b_act`**（通常 bf16），与权重量化精度 `b_w` 无关：权重 INT4 不会让 all-reduce 消息变小。

## 六、Step 3：延迟模型

**TTFT（prefill，算力轴）：**

```
FLOPs_prefill = B * N_in * flops_per_token + 4 * L * N_in^2 * (q_heads * head_dim) * B
                └────────┬───────────────┘   └─────────────────┬─────────────────────┘
                         │                                      └ 注意力二次项（QK^T + AV）：唯一随序列长度平方
                         │                                        的项（q_heads ~= hidden / head_dim，见 3.4）
                         └ 矩阵乘项：总 token 数 × 每 token FLOPs

T_compute = FLOPs_prefill / (N_gpu * Peak_FLOPs * MFU)
                            └────────────┬────────────┘
                                         └ 集群有效算力（卡数 × 单卡峰值 × 利用率）

T_comm = TP all-reduce per layer + PP bubble
TTFT   = T_compute + non_overlapped_comm (+ KV transfer if PD-disaggregated)
```

**TPOT（decode，带宽轴）：**

```
bytes_per_step = W_read_per_step / (t * p) + kv_per_token / (t * p) * S_history * B
                 └───────┬───────┘   └──┬──┘   └──────────────┬──────────────────┘
                         │               │                     └ 每卡要读的历史 KV 总量
                         │               │                       （每 token KV ÷ 分片 × 历史长度 × batch）
                         │               └ 分片因子（权重按 TP × PP 切）
                         └ 每步要读的权重量（dense = 全量；MoE 随 batch 专家覆盖率变化，见下）

T_step = bytes_per_step / (BW_eff * group_size) + T_tp_comm
         └───────┬──────┘   └────────┬─────────┘   └──┬────┘
                 │                    │                 └ TP 通信耗时（见第五节通用式）
                 │                    └ 组内聚合的有效显存带宽
                 └ 每卡每步读取的总字节数

TPOT = ITL = T_step
```

**MoE 注意**：第一项不是全量权重。非专家权重（attention、shared expert、embedding）每步必读；专家权重的读取量取决于 batch 内的专家覆盖率——B 小时只读 top-k 命中的专家，B 增大时趋近全量专家集合。dense 模型无此问题，`W_read_per_step = W_bytes` 恒成立。

**TP 注意**：decode 阶段 TP 的 all-reduce 消息很小（`B × h`），此时通信是延迟受限而非带宽受限——这正是「TP 在小 batch 下反而亏」的来源。而 TP 对 decode 的收益在于聚合多卡 HBM 带宽，两头对冲后是否划算由模型自动算出。

## 七、Step 4：指标汇总

```
E2E_latency = TTFT + N_out * TPOT
              └┬┘   └────┬─────┘
               │          └ decode 总耗时（每步耗时 × 输出长度）
               └ 首 token 延迟（= prefill 总耗时）

throughput = B * N_out / E2E_latency      # output tokens per second
             └────┬────┘   └────┬────┘
                  │              └ 总耗时
                  └ 总产出 token 数（宏观吞吐 = 总产出 ÷ 总耗时）
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
| 量化精度 | 容量轴 + 两阶段时间 | 选择精度同时决定两个量：`b_w`（缩放 `W_bytes`）和 GPU 数据中对应精度的 `Peak_FLOPs` 列（bf16 / fp8 / int4 / fp4）；KV cache 量化由独立的 `b_kv` 控制 |
| PD 分离 | 通信轴 | TTFT 加 KV cache 传输项；prefill / decode 池使用各自独立的并行布局 |
| MoE | 派生常量 | `W_bytes` 含全部专家，`flops_per_token` 用激活参数；EP 变为可选 |

## 九、理想值与校准策略

**第一版输出理想值**：所有效率常数取 1，即理论峰值上界（算力打满、带宽打满、通信完美重叠）。

纯峰值是严格上界，与实测通常有数成差距，因此预留 4~5 个**物理层**效率常数供后续校准：

- `MFU_prefill`：prefill 阶段实际算力利用率
- `BW_eff_decode`：decode 阶段有效显存带宽
- 通信效率（节点内 / 节点间各一个）
- 计算-通信重叠系数

边界说明：校准只接受硬件 / 物理层面的修正，**不引入 vLLM / SGLang 等框架的引擎级校准因子**——它们的稳态吞吐来自 continuous batching 前提，与本模型的单 batch 假设不可比（见附录）。GPU 数据中即便带有实测利用率字段，理想值模式下也不采用。

单节点用实测数据拟合这几个常数即可，参数空间小，校准是良态问题。多节点不追求精确，只额外引入一个关于通信量 / 跳数的校正因子，形式限于线性、最多二次模型（与第一节简化假设 4 对应）。

## 十、技术方案与代码结构

- **语言**：TypeScript，计算核心为纯函数库，与 UI 解耦，可单测
- **部署**：Vercel。全部计算在浏览器端完成，无后端、无运行时成本
- **框架**：Vite + React（或 Next.js 静态导出），待定；计算核心不依赖所选框架
- **数据层**：模型库 / GPU 库从 tps 项目的 ESM JS 数据迁移（`export default` 结构可平滑转 TS），数值使用前需重新校验

```
src/
  core/
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
