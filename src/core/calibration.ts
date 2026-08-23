// Calibration constants.
//
// v1 ships IDEAL (all efficiencies = 1, alpha = 0): pure roofline upper bound.
// Later calibration fits these against single-node measurements; only
// hardware/physical corrections are allowed here, never engine-level factors
// (vLLM / SGLang steady-state numbers are out of scope for this model).

export interface Calibration {
  // Prefill compute utilization (achieved fraction of peak FLOPs).
  mfuPrefill: number;
  // Decode effective HBM bandwidth fraction.
  bwEffDecode: number;
  // Communication efficiency fractions (applied as BW divisors).
  commEffIntra: number;
  commEffInter: number;
  // Fraction of TP all-reduce time overlapped with compute (ideal = 1,
  // i.e. communication fully hidden; 0 = fully exposed on the critical path).
  tpCommOverlap: number;
  // Fraction of EP all-to-all time overlapped with compute (ideal = 1).
  // EP all-to-all is asynchronous point-to-point and typically overlaps
  // better than TP's synchronous per-layer all-reduce.
  epCommOverlap: number;
  // Fraction of PP point-to-point activation transfer overlapped with compute
  // (ideal = 1). Pipeline P2P of successive microbatches overlaps well once
  // the pipeline is full; the fill/drain cost is captured by the bubble factor.
  ppCommOverlap: number;
  // Small-message collective latency, measured total (not per-hop), in ms.
  // nccl-tests reports microseconds: divide by 1000 before filling in.
  // Ideal mode keeps 0 => pure bandwidth upper bound.
  alphaIntraMs: number;
  alphaInterMs: number;
}

export const IDEAL: Calibration = {
  mfuPrefill: 1,
  bwEffDecode: 1,
  commEffIntra: 1,
  commEffInter: 1,
  tpCommOverlap: 1,
  epCommOverlap: 1,
  ppCommOverlap: 1,
  alphaIntraMs: 0,
  alphaInterMs: 0,
};

// Best-guess per-call small-message collective latency (ms) for when the target
// machine has not been micro-benchmarked. Intra (NVLink, ~8 GPUs) ~10 us;
// inter (InfiniBand) ~30 us. Ranges & method: data/benchmarks/README.md.
// Caveat: alpha only contributes when the matching *CommOverlap < 1 (e.g. decode,
// where small-batch collectives are not overlapped). Do NOT derive alpha from
// batch-1 decode TBT (it is contaminated by small-GEMM kernel latency).
export const DEFAULT_ALPHA_INTRA_MS = 0.01;
export const DEFAULT_ALPHA_INTER_MS = 0.03;

// Exposed (non-overlapped) fraction of a communication interval.
// overlap = 1 means fully hidden; overlap = 0 means fully exposed.
export function exposedComm(commTime: number, overlap: number): number {
  return commTime * (1 - overlap);
}
