// i18n: internationalization support
// Locale type and translations dictionary

import { createContext, useContext, useState, useEffect } from 'react';
import { readLocaleFromUrl } from './useUrlParams';

export type Locale = 'en' | 'zh';

const STORAGE_KEY = 'llm-calc-locale';

// Translation dictionary: key → { en: string, zh: string }
const translations = {
  // Header
  'header.title': { en: 'LLM Inference Calculator', zh: 'LLM 推理计算器' },
  'header.subtitle': { en: 'Static performance model · Ideal upper bound', zh: '静态性能模型 · 理想值上界' },

  // Sections
  'section.model_gpu': { en: 'Model & GPU', zh: '模型与 GPU' },
  'section.interconnect': { en: 'Interconnect', zh: '互连' },
  'section.quant_workload': { en: 'Quantization & Workload', zh: '量化与负载' },
  'section.parallel_switches': { en: 'Parallelism & Switches', zh: '并行与开关' },
  'section.speculative': { en: 'Speculative Decoding', zh: '投机采样 (Speculative Decoding)' },
  'section.calibration': { en: 'Calibration (Advanced)', zh: '校准参数（高级）' },

  // Model & GPU
  'label.model': { en: 'Model', zh: '模型' },
  'label.gpu': { en: 'GPU', zh: 'GPU' },
  'label.gpu_count': { en: 'GPU Count', zh: 'GPU 数目' },
  'label.gpus_per_node': { en: 'GPUs per Node', zh: '每节点 GPU 数' },
  'hint.gpus_per_node': { en: 'Defaults to 8 when total GPUs exceed 8', zh: '总GPU数超过 8 时默认 8' },

  // Interconnect
  'label.intra_node': { en: 'Intra-node Interconnect', zh: '节点内互连（intra-node）' },
  'option.intra_auto': { en: 'Auto (by GPU spec: NVLink or PCIe fallback)', zh: '自动（按 GPU 规格：NVLink 或 PCIe 回退）' },
  'label.inter_node': { en: 'Inter-node Interconnect', zh: '节点间互连（inter-node）' },
  'hint.inter_node': { en: 'Only applies in multi-node (cross-node PP / PD KV transfer)', zh: '仅多节点（跨节点 PP / PD KV 传输）时生效' },

  // Quantization & Workload
  'label.quantization': { en: 'Quantization (unified for weights & KV cache)', zh: '量化（权重与 KV cache 统一）' },
  'label.input_length': { en: 'Input Length', zh: '输入长度' },
  'error.exceeds_context': { en: 'Exceeds model max context', zh: '超过模型最大上下文' },
  'label.output_length': { en: 'Output Length', zh: '输出长度' },
  'label.batch_size': { en: 'Steady-state Total Concurrent Requests', zh: '稳态总并发请求数' },
  'label.prefill_ratio_toggle': { en: 'Steady-state Prefill/Decode Contention Model', zh: '稳态 Prefill/Decode 竞争模型' },
  'desc.prefill_ratio_toggle': { en: 'When enabled, accounts for prefill and decode queuing latency on GPU', zh: '启用后考虑 prefill 与 decode 在 GPU 上的排队延迟' },
  'label.prefill_ratio': { en: 'Steady-state Prefill Ratio', zh: '稳态 Prefill 比例' },
  'hint.prefill_ratio': (params: { decodePct: string }) => ({
    en: `Prefill request ratio in steady state, auto-clipped to 0.1–0.9 (decode is ${params.decodePct}%); typical for reasoning models: 0.2`,
    zh: `稳态下 prefill 请求占比，自动 clip 到 0.1–0.9（decode 为 ${params.decodePct}%）；reasoning 模型典型值 0.2`
  }),

  // Parallelism
  'label.dp': { en: 'DP', zh: 'DP' },
  'label.ep': { en: 'EP', zh: 'EP' },
  'hint.ep_disabled': { en: 'EP not supported for Dense models', zh: 'Dense 模型不支持 EP' },
  'note.parallelism': {
    en: 'TP / PP auto-decided by solver (DP ⇒ TP ⇒ EP ⇒ PP, prioritize fitting in VRAM); DP, EP and their product must not exceed total GPUs and must divide it (illegal values auto-snap to nearest valid divisor)',
    zh: 'TP / PP 由求解器自动决定（DP ⇒ TP ⇒ EP ⇒ PP，优先保证显存装下）；DP、EP 及两者乘积均不超过 GPU 总数，且须整除总数（非法值自动吸附到最近的合法约数）'
  },
  'label.flash_attention': { en: 'FlashAttention', zh: 'FlashAttention' },
  'desc.flash_attention': { en: 'When enabled, activation VRAM counted as O(N·h), avoiding N² attention matrix', zh: '开启后激活显存按 O(N·h) 计，避免 N² 注意力矩阵' },
  'label.pd_disagg': { en: 'PD Disaggregation (Prefill-Decode)', zh: 'PD 分离（Prefill-Decode Disaggregation）' },
  'desc.pd_disagg': { en: 'Prefill and Decode use separate GPU pools', zh: 'Prefill 与 Decode 使用独立 GPU 池' },
  'label.prefill_gpus': { en: 'Prefill GPU Count', zh: 'Prefill GPU 数' },
  'label.decode_gpus': { en: 'Decode GPU Count', zh: 'Decode GPU 数' },
  'label.prefill_dp': { en: 'Prefill DP', zh: 'Prefill DP' },
  'hint.prefill_dp': { en: 'Data parallelism for Prefill pool', zh: 'Prefill 池的数据并行度' },
  'label.decode_dp': { en: 'Decode DP', zh: 'Decode DP' },
  'hint.decode_dp': { en: 'Data parallelism for Decode pool (high DP can maximize TP)', zh: 'Decode 池的数据并行度（高 DP 可最大化 TP）' },
  'note.pd_constraint': (params: { total: number }) => ({
    en: `Sum of two pools must be ≤ total GPUs (${params.total}), some GPUs may be unallocated. DP must divide pool GPU count.`,
    zh: `两池之和须 ≤ GPU 总数（${params.total}），允许部分 GPU 不分配。DP 须整除对应池的 GPU 数。`
  }),
  'label.kv_overlap': (params: { value: string }) => ({
    en: `KV Transfer Overlap Coefficient — ${params.value}`,
    zh: `KV 传输重叠系数 — ${params.value}`
  }),
  'hint.kv_overlap': {
    en: 'Ratio of KV transfer time overlapped (hidden) with prefill: 0 = fully exposed in TTFT, 1 = fully hidden; modern engines with per-layer pipeline ≈ 0.8–1',
    zh: 'KV 传输时间与 prefill 计算重叠（被隐藏）的比例：0 = 完全串行暴露在 TTFT 中，1 = 完全隐藏；有逐层流水的现代引擎约 0.8 ~ 1'
  },

  // Speculative Decoding
  'label.sd_toggle': { en: 'Enable Speculative Decoding', zh: '启用投机采样' },
  'desc.sd_toggle': { en: 'Small draft model + large model verification, reduces TPOT', zh: '小模型草稿 + 大模型验证，降低 TPOT' },
  'label.draft_model': { en: 'Draft Model', zh: '草稿模型（Draft Model）' },
  'placeholder.select_draft': { en: 'Select draft model', zh: '选择草稿模型' },
  'placeholder.no_draft': { en: 'No draft models available', zh: '无可用草稿模型' },
  'error.no_draft_model': (params: { minB: number, maxB: number }) => ({
    en: `No suitable draft model in catalog for current main model (need ${params.minB}B–${params.maxB}B dense model)`,
    zh: `模型库中没有适合当前主模型的草稿模型（需要 ${params.minB}B ~ ${params.maxB}B 的 dense 模型）`
  }),
  'label.draft_tp': { en: 'Draft TP', zh: 'Draft TP' },
  'hint.draft_tp': { en: 'Tensor parallelism for draft model (TP only, defaults to main model TP)', zh: '草稿模型的张量并行度（仅支持 TP，默认跟随主模型 TP）' },
  'label.gamma': { en: 'γ (Draft Steps)', zh: 'γ（草稿步数）' },
  'hint.gamma': { en: 'Tokens generated by draft model before each verification, typical 4–8', zh: '每次验证前草稿模型生成的 token 数，典型 4–8' },
  'label.acceptance_rate': { en: 'Acceptance Rate', zh: '接受率' },
  'hint.acceptance_rate': { en: 'Probability draft token accepted by main model; depends on model pair, typically 0.5–0.9', zh: '草稿 token 被主模型接受的概率；取决于模型配对，通常 0.5–0.9' },
  'note.sd_constraint': (params: { maxDraftTp: number, disaggOn: boolean }) => ({
    en: `Draft and Main models both in GPU memory; Draft only supports TP, not PP/EP/DP. Draft TP defaults to main model TP, max = ${params.disaggOn ? 'Decode' : ''} total GPUs (${params.maxDraftTp}).`,
    zh: `Draft 和 Main 模型同时在 GPU 内存中；Draft 仅支持 TP 并行，不支持 PP/EP/DP。Draft TP 默认跟随主模型 TP，最大值 = ${params.disaggOn ? 'Decode' : ''} GPU 总数（${params.maxDraftTp}）。`
  }),

  // Calibration
  'summary.calibration': { en: 'Default: calibrated values (calibration/README.md anchor)', zh: '默认：校准值（calibration/README.md 锚点）' },
  'label.headroom': { en: 'VRAM Headroom', zh: '显存预留 headroom' },
  'label.mfu': { en: 'MFU (Model FLOPs Utilization)', zh: '算力利用率 MFU' },
  'hint.mfu': { en: 'Model FLOPs Utilization, ratio of peak FLOPs achieved in Prefill phase', zh: 'Model FLOPs Utilization, Prefill 阶段实际达到的峰值算力比例' },
  'label.bw_eff': { en: 'BW_eff (Effective Bandwidth)', zh: '带宽利用率 BW_eff' },
  'hint.bw_eff': { en: 'Effective Bandwidth, ratio of HBM bandwidth achieved in Decode phase', zh: 'Effective Bandwidth, Decode 阶段实际达到的 HBM 显存带宽比例' },
  'label.comm_eff_intra': { en: 'Communication Efficiency (Intra-node)', zh: '通信效率（节点内）' },
  'label.comm_eff_inter': { en: 'Communication Efficiency (Inter-node)', zh: '通信效率（节点间）' },
  'label.tp_comm_overlap': { en: 'TP Communication Overlap', zh: 'TP 通信重叠' },
  'label.ep_comm_overlap': { en: 'EP Communication Overlap', zh: 'EP 通信重叠' },
  'label.pp_comm_overlap': { en: 'PP Communication Overlap', zh: 'PP 通信重叠' },
  'label.alpha_intra': { en: 'α Intra-node (ms)', zh: 'α 节点内 (ms)' },
  'label.alpha_inter': { en: 'α Inter-node (ms)', zh: 'α 节点间 (ms)' },
  'btn.reset_ideal': { en: 'Reset to Ideal', zh: '重置为理想值' },
  'title.reset_ideal': { en: 'Upper bound: all efficiencies = 1, only VRAM headroom stays at 0.1', zh: '性能上界：全部效率 = 1，仅显存预留保持 0.1' },
  'btn.reset_calibrated': { en: 'Reset to Calibrated', zh: '重置为校准值' },
  'title.reset_calibrated': { en: 'Max values per calibration/README.md anchor', zh: '按 calibration/README.md 锚点取最大值' },

  // Results
  'error.cannot_compute': { en: 'Cannot Compute', zh: '无法计算' },
  'error.oom': (params: { overBytes: string }) => ({
    en: `VRAM Overflow (exceeds per-GPU by ${params.overBytes})`,
    zh: `显存不足（单卡超出 ${params.overBytes}）`
  }),
  'hint.oom': { en: 'Try lower quantization, more GPUs, higher EP/PP, or reduce batch / sequence length', zh: '试试更低量化、更多 GPU、更高 EP/PP，或减小 batch / 序列长度' },
  'status.deployable': { en: 'Deployable', zh: '可部署' },
  'label.loading_chart': { en: 'Loading chart…', zh: '加载图表…' },
  'label.invalid_config': { en: 'Invalid Configuration', zh: '无效配置' },

  // Metric tiles
  'metric.ttft': { en: 'TTFT (first token)', zh: 'TTFT（首 token）' },
  'metric.tpot': { en: 'TPOT (per token)', zh: 'TPOT（每 token）' },
  'metric.e2e': { en: 'End-to-end Latency', zh: '端到端延迟' },
  'metric.throughput': { en: 'System Throughput', zh: '系统吞吐' },

  // Layout card
  'title.layout_pd': { en: 'Parallel Layout (PD Disaggregation)', zh: '并行布局（PD 分离）' },
  'label.prefill_pool': { en: 'Prefill Pool', zh: 'Prefill 池' },
  'label.decode_pool': { en: 'Decode Pool', zh: 'Decode 池' },
  'note.no_feasible_layout': { en: 'No feasible layout; closest reference below.', zh: '没有可行布局；以下为最接近的参照。' },
  'note.no_layout': { en: 'No layout available.', zh: '无可用布局。' },
  'title.layout': { en: 'Parallel Layout', zh: '并行布局' },
  'note.no_layout_fits': { en: 'No layout fits current config; closest reference below.', zh: '没有任何布局能装下当前配置；以下为最接近的参照。' },
  'note.layout_constraint': (params: { gpusPerNode: number }) => ({
    en: `Candidate layouts constrained by "TP ≤ GPUs per node (${params.gpusPerNode})", sorted by TP ⇒ EP ⇒ PP`,
    zh: `候选布局受「TP ≤ 每节点 GPU 数（${params.gpusPerNode}）」约束，按 TP ⇒ EP ⇒ PP 排序`
  }),
  'label.optimal_gpu_alloc': { en: 'Optimal GPU Allocation', zh: '最优 GPU 分配' },
  'note.optimal_current': { en: ' — already optimal ✓', zh: ' — 当前已最优 ✓' },
  'note.suggest_prefill': (params: { optGpus: number }) => ({
    en: ` — suggest adjusting Prefill GPU count to ${params.optGpus}`,
    zh: ` — 建议调整 Prefill GPU 数为 ${params.optGpus}`
  }),
  'note.pipeline_balance': { en: 'Pipeline balancing: Prefill output rate = Decode consumption rate, preventing node starvation or backlog', zh: '流水线配平：Prefill 产出速率 = Decode 消耗速率，防止节点饥饿或积压' },

  // VRAM card
  'title.vram': { en: 'VRAM Usage (per GPU)', zh: '显存占用（每卡）' },
  'seg.weights': { en: 'Weights', zh: '权重' },
  'seg.kv': { en: 'KV cache', zh: 'KV cache' },
  'seg.activation': { en: 'Activation', zh: '激活' },
  'seg.overhead': { en: 'Overhead', zh: '开销' },
  'seg.draft_weights': { en: 'Draft Weights', zh: 'Draft 权重' },
  'seg.draft_kv': { en: 'Draft KV', zh: 'Draft KV' },
  'label.vram_usage': (params: { total: string, cap: string, pct: string }) => ({
    en: `${params.total} / ${params.cap} (${params.pct})`,
    zh: `${params.total} / ${params.cap}（${params.pct}）`
  }),
  'label.over_capacity': { en: ' — exceeds', zh: ' — 超出' },
  'label.max_batch_full': (params: { poolBMax: string }) => ({
    en: `Max Batch size: ${params.poolBMax}`,
    zh: `最大 Batch size：${params.poolBMax}`
  }),
  'label.system_max_batch': (params: { systemBMax: string }) => ({
    en: ` | System max: ${params.systemBMax}`,
    zh: ` | 系统最大：${params.systemBMax}`
  }),
  'note.vram_capacity': {
    en: 'Capacity accounts for headroom. Max batch assumes requests evenly distributed across generation (avg inputLen + outputLen/2).',
    zh: '容量已扣除 headroom 预留。最大 Batch 假设请求均匀分布在生成过程中（平均 inputLen + outputLen/2）。'
  },
  'label.system_total_max_batch': (params: { totalSystemBMax: string }) => ({
    en: `System total max Batch size: ${params.totalSystemBMax} = Min(Prefill/r, Decode/(1−r))`,
    zh: `系统总最大 Batch size：${params.totalSystemBMax} = Min(Prefill/r, Decode/(1−r))`
  }),
  'label.prefill_pool_batch': (params: { batch: number }) => ({
    en: `Prefill Pool (current batch ${params.batch})`,
    zh: `Prefill 池（当前 batch ${params.batch}）`
  }),
  'label.decode_pool_batch': (params: { batch: number }) => ({
    en: `Decode Pool (current batch ${params.batch})`,
    zh: `Decode 池（当前 batch ${params.batch}）`
  }),

  // Phase card
  'title.phase': { en: 'Phase Details', zh: '阶段明细' },
  'phase.prefill': { en: 'Prefill (compute-bound)', zh: 'Prefill（计算受限）' },
  'phase.decode': { en: 'Decode (bandwidth-bound)', zh: 'Decode（带宽受限）' },
  'label.total_flops': { en: 'Total FLOPs', zh: '总 FLOPs' },
  'label.compute_time': { en: 'Compute Time', zh: '计算时间' },
  'label.comm_time_total': { en: 'Communication Time (total)', zh: '通信时间（总）' },
  'label.throughput_input': { en: 'Throughput (input token/s)', zh: '吞吐（输入 token/s）' },
  'label.kv_transfer_exposed': { en: 'KV Transfer (exposed)', zh: 'KV 传输（暴露）' },
  'label.compute_util': { en: 'Compute Utilization', zh: '算力利用率' },
  'label.weight_read': { en: 'Weight Read / GPU·step', zh: '权重读取 / 卡·步' },
  'label.kv_read': { en: 'KV Read / GPU·step', zh: 'KV 读取 / 卡·步' },
  'label.bandwidth_time': { en: 'Bandwidth Time', zh: '带宽时间' },
  'label.expert_coverage': { en: 'Expert Coverage', zh: '专家覆盖率' },
  'label.throughput_output': { en: 'Throughput (output token/s)', zh: '吞吐（输出 token/s）' },
  'label.bandwidth_util': { en: 'Bandwidth Utilization', zh: '带宽利用率' },
  'note.comm_overlap': {
    en: 'Note: "Communication Time (total)" is raw communication; what enters latency is the exposed portion = total × (1 − overlap coefficient). When overlap = 1 (ideal), communication is fully hidden by compute, TTFT / TPOT exclude communication; lower TP / EP / PP communication overlap in Calibration panel to see exposed cost.',
    zh: '注：「通信时间（总）」为原始通信量，进入延迟的是其暴露部分 = 总量 ×（1 − 重叠系数）。重叠系数 = 1（理想模式）时通信被计算完全隐藏，TTFT / TPOT 不含通信；可在「校准参数」面板调低 TP / EP / PP 通信重叠查看暴露的通信代价。'
  },
  'note.steady_state_split': (params: { Bp: number, Bd: number, total: number }) => ({
    en: `Steady-state workload split: Prefill batch = ${params.Bp}, Decode batch = ${params.Bd} (split by input ratio from total concurrency ${params.total}).`,
    zh: `稳态负载分配：Prefill batch = ${params.Bp}，Decode batch = ${params.Bd}（按输入比例拆分总并发 ${params.total}）。`
  }),

  // Speculative card
  'title.speculative': { en: 'Speculative Decoding Details', zh: '投机采样明细' },
  'label.draft_model_info': { en: 'Draft Model', zh: '草稿模型' },
  'label.model_name': { en: 'Model', zh: '模型' },
  'label.expected_tokens': { en: 'Expected tokens per cycle', zh: '每周期期望 token' },
  'label.timing_analysis': { en: 'Timing Analysis', zh: '时序分析' },
  'label.draft_step_time': { en: 'Draft step time', zh: '草稿步时间' },
  'label.verify_step_time': { en: 'Verify step time', zh: '验证步时间' },
  'label.cycle_time': { en: 'Cycle time', zh: '周期时间' },
  'label.tpot_speculative': { en: 'TPOT (speculative)', zh: 'TPOT（投机）' },
  'label.tpot_baseline': { en: 'TPOT (baseline)', zh: 'TPOT（基线）' },
  'label.speedup': { en: 'Speedup', zh: '加速比' },
  'note.speculative': {
    en: 'Speculative decoding uses a small draft model to generate candidate tokens, then the large model verifies in parallel, reducing per-token generation time (TPOT). Draft and Main models both in GPU memory; Draft only supports TP.',
    zh: '投机采样通过小模型（Draft）生成候选 token，大模型（Main）并行验证，从而降低每 token 生成时间（TPOT）。Draft 和 Main 模型同时在 GPU 内存中；Draft 仅支持 TP 并行。'
  },

  // Batch sweep chart
  'title.batch_sweep': { en: 'Throughput & Latency vs Batch Size', zh: 'Throughput & Latency vs Batch Size' },
  'note.no_feasible_batch': { en: 'No feasible batch under current config.', zh: '当前配置下没有可行的 batch。' },
  'label.batch_size_axis': { en: 'Batch Size', zh: 'Batch Size' },
  'label.throughput_axis': { en: 'Throughput', zh: 'Throughput' },
  'label.latency_axis': { en: 'Latency', zh: 'Latency' },
  'label.throughput_tps': { en: 'Throughput (tok/s)', zh: '吞吐 (tok/s)' },
  'label.e2e_latency': { en: 'End-to-end Latency', zh: '端到端延迟' },
  'note.oom_batch': (params: { batches: string }) => ({
    en: `batch = ${params.batches} VRAM overflow (exceeds max Batch size), not shown in chart.`,
    zh: `batch = ${params.batches} 显存不足（超过最大 Batch size），未在图中显示。`
  }),
  'label.oom_tooltip': { en: 'batch = : VRAM overflow', zh: 'batch = ：显存不足' },
  'label.throughput_tooltip': (params: { tps: string }) => ({
    en: `Throughput: ${params.tps} tok/s`,
    zh: `吞吐：${params.tps} tok/s`
  }),
  'label.e2e_tooltip': (params: { ms: string }) => ({
    en: `End-to-end latency: ${params.ms}`,
    zh: `端到端延迟：${params.ms}`
  }),

  // Search select
  'placeholder.search': { en: 'Search…', zh: '搜索…' },
  'label.all': { en: 'All', zh: 'All' },
  'label.copy': { en: 'Copy', zh: '复制' },
  'note.more_items': (params: { count: number }) => ({
    en: `… ${params.count} more items, please filter by keyword`,
    zh: `… 其余 ${params.count} 项，请输入关键词过滤`
  }),
  'note.no_match': { en: 'No match', zh: '无匹配项' },

  // Warnings
  'warn.input_exceeds_ctx': (params: { inputLen: number, maxCtx: number, ctxStr: string }) => ({
    en: `Input length ${params.inputLen} exceeds model max context ${params.maxCtx} (${params.ctxStr})`,
    zh: `输入长度 ${params.inputLen} 超过模型最大上下文 ${params.maxCtx}（${params.ctxStr}）`
  }),
  'warn.pd_min_gpus': { en: 'PD disaggregation requires at least 2 GPUs', zh: 'PD 分离至少需要 2 张 GPU' },
  'warn.pd_exceeds_total': (params: { pN: number, dN: number, total: number }) => ({
    en: `Prefill (${params.pN}) + Decode (${params.dN}) GPU sum ${params.pN + params.dN} exceeds total ${params.total}`,
    zh: `Prefill（${params.pN}）+ Decode（${params.dN}）GPU 数之和 ${params.pN + params.dN} 超过总数 ${params.total}`
  }),

  // Dynamic model sub labels
  'model.sub.moe': (params: { activeB: number }) => ({
    en: `active ${params.activeB}B`,
    zh: `激活 ${params.activeB}B`
  }),
  'model.sub.draft': (params: { draftB: number, mainB: number, pct: number }) => ({
    en: `${params.draftB}B (${params.pct}% of main ${params.mainB}B)`,
    zh: `${params.draftB}B（主模型 ${params.mainB}B 的 ${params.pct}%）`
  }),
} as const;

export type TranslationKey = keyof typeof translations;

// Locale context
const LocaleContext = createContext<{
  locale: Locale;
  setLocale: (locale: Locale) => void;
}>({
  locale: 'en',
  setLocale: () => {},
});

// Provider component
export function LocaleProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(() => {
    // URL takes precedence over localStorage for shareable links.
    const fromUrl = readLocaleFromUrl();
    if (fromUrl === 'zh' || fromUrl === 'en') return fromUrl;
    const stored = localStorage.getItem(STORAGE_KEY);
    return (stored === 'zh' ? 'zh' : 'en') as Locale;
  });

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, locale);
  }, [locale]);

  const setLocale = (newLocale: Locale) => {
    setLocaleState(newLocale);
  };

  return (
    <LocaleContext.Provider value={{ locale, setLocale }}>
      {children}
    </LocaleContext.Provider>
  );
}

// Hook to use i18n
export function useI18n() {
  const { locale, setLocale } = useContext(LocaleContext);

  const t = (key: TranslationKey, params?: Record<string, any>): string => {
    const entry = translations[key];
    if (!entry) {
      console.warn(`Missing translation key: ${key}`);
      return key;
    }

    // If entry is a function, call it with params
    if (typeof entry === 'function') {
      const result = (entry as any)(params || {});
      return result[locale] || result.en;
    }

    // Otherwise it's a static object
    return (entry as any)[locale] || (entry as any).en;
  };

  return { locale, setLocale, t };
}
