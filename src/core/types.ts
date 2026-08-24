// Shared types for the inference calculator core.

export type QuantPrecision = 'fp32' | 'fp16' | 'bf16' | 'fp8' | 'int8' | 'int4' | 'fp4';

// Bytes per parameter under each quantization precision.
export const BYTES_PER_PARAM: Record<QuantPrecision, number> = {
  fp32: 4,
  fp16: 2,
  bf16: 2,
  fp8: 1,
  int8: 1,
  int4: 0.5,
  fp4: 0.5,
};

// Activation / communication message precision is independent of weight
// quantization; weights at INT4 still all-reduce bf16 activations.
export const B_ACT = 2;

export type MoeExecution = 'shared_routed' | 'top1_routed' | 'parallel_dense_routed';

export interface MoeSpec {
  experts: number;
  expertsPerToken: number; // top-k routed experts per token
  activeParamsB: number; // active params per token, in billions
  execution: MoeExecution;
}

export interface ModelSpec {
  id: string;
  name: string;
  type: 'dense' | 'moe';
  paramsB: number; // total params in billions (MoE: all experts included)
  layers: number;
  hiddenSize: number;
  kvHeads: number; // KV heads of standard / local layers
  headDim: number;
  qHeads?: number; // query heads; falls back to hiddenSize / headDim when absent
  moe?: MoeSpec;
  // --- KV-cache shape variants (all optional; absent = standard GQA/MHA) ---
  mlaRatio?: number; // MLA compression factor applied to KV (e.g. DeepSeek V3 ~0.0176)
  linearAttentionLayers?: number; // layers contributing zero standard KV cache
  mambaRatio?: number; // fraction of SSM layers (constant state, no growing KV)
  localLayers?: number; // sliding-window layers (paired with slidingWindow > 0)
  slidingWindow?: number; // per-layer KV token cap for local layers
  globalKvHeads?: number; // KV heads of global layers when they differ from local
  globalHeadDim?: number;
  maxCtx: number;
  hf?: string; // HuggingFace model URL
}

export interface GpuSpec {
  id: string;
  name: string;
  vramGb: number;
  bwGbps: number; // HBM bandwidth
  peakTflops: Partial<Record<QuantPrecision, number>>; // dense peak per precision
  nvlinkBwGbps?: number; // intra-node link bandwidth; undefined => PCIe
}

export interface Interconnect {
  intraNodeBwGbps: number; // resolved from GpuSpec.nvlinkBwGbps or PCIe fallback
  interNodeBwGbps: number; // user input (InfiniBand etc.)
}

export interface ParallelLayout {
  tp: number;
  pp: number;
  ep: number; // must be 1 for dense models
  dp: number;
}

// Prefill-Decode disaggregation: two independent GPU pools.
export interface PdDisaggConfig {
  prefillGpus: number;
  decodeGpus: number;
  prefillLayout: ParallelLayout;
  decodeLayout: ParallelLayout;
  // Fraction of KV transfer time overlapped with prefill compute (ideal = 1).
  kvTransferOverlap: number;
}

// Speculative decoding: a small draft model proposes γ tokens that the main
// model verifies in a single parallel forward pass. Both models reside in GPU
// memory simultaneously; they alternate on the same GPU cluster.
export interface SpeculativeConfig {
  draftModel: ModelSpec;
  draftTp: number; // 1 ≤ draftTp ≤ numGpus (or decodeGpus if PD disagg)
  gamma: number; // draft steps per verification round (typically 4-8)
  acceptanceRate: number; // user-supplied, 0..1
}

export interface Workload {
  batchSize: number;
  inputLen: number;
  outputLen: number;
  // Steady-state workload composition: fraction of requests in prefill.
  // Decode fraction is (1 - prefillRatio). Absent defaults to 0.5 (equal split).
  // Typical reasoning model: ~0.2 (prefill 20%, decode 80%).
  prefillRatio?: number;
}

export interface SystemSpec {
  model: ModelSpec;
  gpu: GpuSpec;
  gpusPerNode: number;
  interNodeBwGbps: number;
  // Optional intra-node bandwidth override (e.g. an explicit NVLink/PCIe
  // choice from the UI). Absent => resolved from GpuSpec.nvlinkBwGbps or
  // the PCIe fallback.
  intraNodeBwGbps?: number;
  workload: Workload;
  weightQuant: QuantPrecision;
  kvQuant: QuantPrecision; // KV cache precision, independent of weight quant
  layout: ParallelLayout;
  flashAttention: boolean;
  headroom: number; // reserved VRAM fraction, e.g. 0.1
  disagg?: PdDisaggConfig;
  speculative?: SpeculativeConfig; // speculative decoding configuration
  // Internal hint used by the layout solver: when evaluating a single layout
  // for a PD pool (without a full disagg config), tells evaluate() which
  // pool-specific seqLen to use for VRAM sizing. Never set by the UI.
  solverPdMode?: 'prefill' | 'decode';
}

export const GB = 1e9;
export const GiB = 2 ** 30;

// Default steady-state prefill ratio: 0.2 for reasoning models (prefill ~20%,
// decode ~80% of steady-state requests). Used when Workload.prefillRatio is absent.
export const DEFAULT_PREFILL_RATIO = 0.2;
