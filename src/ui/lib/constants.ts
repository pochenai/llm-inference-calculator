// UI-level defaults and display thresholds (single source of truth).

import type { QuantPrecision } from '../../core/types';
import type { Calibration } from '../../core/calibration';

// --- GPU-count display thresholds ---
// Per-node GPU count and inter-node interconnect only matter once the cluster
// can actually span nodes; PD disaggregation needs at least two pools.
export const MIN_GPUS_FOR_PER_NODE = 4; // show the per-node GPU count input
export const MIN_GPUS_FOR_INTER_NODE = 4; // show the inter-node interconnect select
export const MIN_GPUS_FOR_PD_DISAGG = 2; // show the PD-disaggregation toggle

// --- per-node defaults ---
// 8 is the de-facto standard node size (NVIDIA HGX/DGX baseboards).
// The UI default is min(total GPUs, 8), re-applied on every GPU-count change.
export const DEFAULT_GPUS_PER_NODE = 8;

// --- PD disaggregation ---
// Fraction of the KV transfer hidden by overlap with prefill compute.
// Modern engines pipeline the transfer layer-by-layer, so ~0.8-1 is realistic.
export const DEFAULT_KV_TRANSFER_OVERLAP = 0.8;

// --- calibrated preset (the UI default) ---
// Anchors from calibration/README.md (v0, order-of-magnitude), taking the
// maximum where a range is given:
//   mfuPrefill   0.6   anchors ~0.17 (llama.cpp) / ~0.5 (TRT-LLM) -> max
//   bwEffDecode  0.55  anchor ~0.53
//   *CommOverlap 0.5   README "conservative start ≈ 0.5"
//   alpha        README defaults when unmeasured
// commEff* have no README anchor yet; 0.9 reflects typical NCCL
// large-message efficiency until nccl-tests numbers replace it.
export const CALIBRATED_PRESET: Calibration = {
  mfuPrefill: 0.6,
  bwEffDecode: 0.55,
  commEffIntra: 0.9,
  commEffInter: 0.9,
  tpCommOverlap: 0.5,
  epCommOverlap: 0.5,
  ppCommOverlap: 0.5,
  alphaIntraMs: 0.01,
  alphaInterMs: 0.03,
};

// --- memory ---
export const DEFAULT_HEADROOM = 0.1; // reserved VRAM fraction

// --- interconnect fallback ---
export const DEFAULT_INTER_NODE_BW_GBPS = 50; // InfiniBand NDR

// --- workload defaults ---
export const DEFAULT_BATCH_SIZE = 1;
export const DEFAULT_INPUT_LEN = 1024;
export const DEFAULT_OUTPUT_LEN = 1024;
export const DEFAULT_QUANT: QuantPrecision = 'bf16';
// Re-export from core (single source of truth for calculation defaults).
export { DEFAULT_PREFILL_RATIO } from '../../core/types';

// --- speculative decoding ---
// Draft model size range relative to main model (5-10x smaller).
export const SD_RATIO_MIN = 0.1; // 10x smaller
export const SD_RATIO_MAX = 0.2; // 5x smaller
export const DEFAULT_GAMMA = 5; // draft steps per verification round
export const DEFAULT_ACCEPTANCE_RATE = 0.7; // typical acceptance rate
