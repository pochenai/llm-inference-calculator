// UI-level defaults and display thresholds (single source of truth).

import type { QuantPrecision } from '../../core/types';

// --- GPU-count display thresholds ---
// Per-node GPU count and inter-node interconnect only matter once the cluster
// can actually span nodes; PD disaggregation needs at least two pools.
export const MIN_GPUS_FOR_PER_NODE = 4; // show the per-node GPU count input
export const MIN_GPUS_FOR_INTER_NODE = 4; // show the inter-node interconnect select
export const MIN_GPUS_FOR_PD_DISAGG = 2; // show the PD-disaggregation toggle

// --- per-node defaults ---
// 8 is the de-facto standard node size (NVIDIA HGX/DGX baseboards).
export const DEFAULT_GPUS_PER_NODE = 8;
// Once the cluster is larger than this, gpus-per-node defaults back to 8.
export const AUTO_PER_NODE_ABOVE = 8;

// --- PD disaggregation ---
// Fraction of the KV transfer hidden by overlap with prefill compute.
// Modern engines pipeline the transfer layer-by-layer, so ~0.8-1 is realistic.
export const DEFAULT_KV_TRANSFER_OVERLAP = 0.8;

// --- memory ---
export const DEFAULT_HEADROOM = 0.1; // reserved VRAM fraction

// --- interconnect fallback ---
export const DEFAULT_INTER_NODE_BW_GBPS = 50; // InfiniBand NDR

// --- workload defaults ---
export const DEFAULT_BATCH_SIZE = 1;
export const DEFAULT_INPUT_LEN = 1024;
export const DEFAULT_OUTPUT_LEN = 1024;
export const DEFAULT_QUANT: QuantPrecision = 'bf16';
