// ModelSpec -> derived constants (W_bytes, kv geometry, flops_per_token).

import type { ModelSpec, QuantPrecision } from './types.js';
import { BYTES_PER_PARAM } from './types.js';
import { CalcError } from './errors.js';

export interface KvGeometry {
  // Marginal KV bytes per added token (sliding layers already at window cap).
  bytesPerToken: number;
  // Total KV bytes for one sequence of the given length.
  totalBytes: (seqLen: number) => number;
  // Layers with standard quadratic attention (global + local/sliding-window).
  fullAttentionLayers: number;
  globalLayers: number;
  localLayers: number;
  linearLayers: number;
  ssmLayers: number;
  slidingWindow: number; // 0 when none
}

// Partition layers into KV-shape groups. Ordering does not matter because
// only aggregate sums enter the model.
export function buildKvGeometry(model: ModelSpec, bKv: number): KvGeometry {
  const window = model.slidingWindow ?? 0;
  // tps convention: local_layers + sliding_window === 0 encodes linear attention.
  const linearLayers =
    model.linearAttentionLayers ??
    (model.localLayers != null && window === 0 ? model.localLayers : 0);
  const ssmLayers = Math.round((model.mambaRatio ?? 0) * model.layers);
  const localLayers = model.localLayers != null && window > 0 ? model.localLayers : 0;
  const globalLayersRaw = model.layers - linearLayers - ssmLayers - localLayers;
  if (globalLayersRaw < 0) {
    throw new CalcError(
      'contradictory-layers',
      `Model '${model.id}' has contradictory layer counts: ` +
        `layers=${model.layers} but linearAttentionLayers=${linearLayers} + ` +
        `ssmLayers=${ssmLayers} (mambaRatio=${model.mambaRatio ?? 0}) + ` +
        `localLayers=${localLayers} sums to ${linearLayers + ssmLayers + localLayers}, ` +
        `leaving ${globalLayersRaw} global layers. ` +
        `Special layers must not exceed total layers.`,
    );
  }
  const globalLayers = globalLayersRaw;

  const compress = model.mlaRatio ?? 1;
  const gKvHeads = model.globalKvHeads ?? model.kvHeads;
  const gHeadDim = model.globalHeadDim ?? model.headDim;

  // Elements per token, single side (K or V), per layer group.
  const globalElems = gKvHeads * gHeadDim;
  const localElems = model.kvHeads * model.headDim;

  const bytesPerToken = 2 * bKv * compress * (globalLayers * globalElems + localLayers * localElems);

  const totalBytes = (seqLen: number): number => {
    if (seqLen < 0) {
      throw new CalcError(
        'negative-seqlen',
        `totalBytes called with negative seqLen=${seqLen}. ` +
          `Sequence length must be non-negative; this is a caller bug.`,
      );
    }
    const s = seqLen;
    const globalPart = globalLayers * globalElems * s;
    const localPart = localLayers * localElems * Math.min(s, window);
    return 2 * bKv * compress * (globalPart + localPart);
  };

  return {
    bytesPerToken,
    totalBytes,
    fullAttentionLayers: globalLayers + localLayers,
    globalLayers,
    localLayers,
    linearLayers,
    ssmLayers,
    slidingWindow: window,
  };
}

// Number of query heads. q_heads is absent from the dataset; fall back to
// hidden / head_dim (see README 3.4).
export function qHeadsOf(model: ModelSpec): number {
  return model.qHeads ?? Math.max(1, Math.round(model.hiddenSize / model.headDim));
}

// Query-side dimension used by the attention quadratic FLOP term.
export function qDimOf(model: ModelSpec): number {
  return qHeadsOf(model) * model.headDim;
}

// Quadratic attention FLOPs for one sequence of length N, all layers:
//   global layers: 4 * N^2 * qDim   (QK^T + AV)
//   local layers:  4 * N * min(N, window) * qDim
//   linear / SSM layers contribute 0.
export function attentionQuadFlops(model: ModelSpec, kv: KvGeometry, seqLen: number): number {
  const qDim = qDimOf(model);
  if (seqLen < 0) {
    throw new CalcError(
      'negative-seqlen',
      `attentionQuadFlops called with negative seqLen=${seqLen}. ` +
        `Sequence length must be non-negative; this is a caller bug.`,
    );
  }
  const n = seqLen;
  const globalPart = kv.globalLayers * n * n;
  const localPart =
    kv.localLayers * n * (kv.slidingWindow > 0 ? Math.min(n, kv.slidingWindow) : n);
  return 4 * qDim * (globalPart + localPart);
}

// Non-expert params of a MoE model (weights not sharded by EP, i.e. attention,
// shared experts, embeddings). Solved from:
//   P = N + experts * per_expert,  A = N + k * per_expert
//   => N = (P*k - experts*A) / (k - experts)
export function nonExpertParamsB(model: ModelSpec): number {
  if (model.type !== 'moe' || !model.moe) return model.paramsB;
  const { experts, expertsPerToken: k, activeParamsB } = model.moe;
  if (k >= experts) return activeParamsB; // top-all degenerates to dense
  const denom = k - experts;
  const n = (model.paramsB * k - experts * activeParamsB) / denom;
  if (!Number.isFinite(n) || n < 0 || n > activeParamsB) {
    // The algebraic solution is non-finite, negative, or exceeds activeParamsB,
    // meaning the MoE data (paramsB, activeParamsB, experts, expertsPerToken)
    // is self-contradictory. Refuse to invent a plausible-looking number.
    throw new CalcError(
      'inconsistent-moe',
      `Model '${model.id}' has inconsistent MoE parameters: ` +
        `paramsB=${model.paramsB}, activeParamsB=${activeParamsB}, ` +
        `experts=${experts}, expertsPerToken=${k}. ` +
        `Solved nonExpertParamsB=${n} (expected 0 <= N <= ${activeParamsB}). ` +
        `Check that paramsB, activeParamsB, experts, and expertsPerToken are consistent.`,
    );
  }
  return n;
}

export interface DerivedConstants {
  totalParams: number; // raw param count (MoE: all experts)
  activeParams: number; // active per token
  nonExpertParams: number; // not sharded by EP
  expertParams: number; // routed experts, sharded by EP
  wBytesTotal: number;
  wNonexpertBytes: number;
  wExpertBytes: number;
  flopsPerToken: number;
  kv: KvGeometry;
  qHeads: number; // number of query heads
  qDim: number; // qHeads * headDim
}

export function deriveConstants(
  model: ModelSpec,
  weightQuant: QuantPrecision,
  kvQuant: QuantPrecision,
): DerivedConstants {
  const bW = BYTES_PER_PARAM[weightQuant];
  const bKv = BYTES_PER_PARAM[kvQuant];

  const totalParams = model.paramsB * 1e9;
  const activeParams = (model.moe?.activeParamsB ?? model.paramsB) * 1e9;
  const nonExpertParams = nonExpertParamsB(model) * 1e9;
  const expertParams = totalParams - nonExpertParams;

  return {
    totalParams,
    activeParams,
    nonExpertParams,
    expertParams,
    wBytesTotal: totalParams * bW,
    wNonexpertBytes: nonExpertParams * bW,
    wExpertBytes: expertParams * bW,
    flopsPerToken: 2 * activeParams,
    kv: buildKvGeometry(model, bKv),
    qHeads: qHeadsOf(model),
    qDim: qDimOf(model),
  };
}
