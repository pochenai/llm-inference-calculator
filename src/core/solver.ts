// Automatic parallel-layout resolution.
//
// The user fixes the total GPU count plus DP and EP (both default 1); the
// solver then decides TP and PP. Decision order follows DP => TP => EP => PP:
//
//   1. DP and EP come from user input and are sanitized against the GPU
//      count and model type (clamped to the nearest valid value, with a
//      human-readable note when that happens).
//   2. TP candidates are the divisors of the remaining GPUs per DP replica,
//      capped by gpusPerNode, tried from largest to smallest. PP takes
//      whatever remains (TP x PP x EP x DP = N_gpu always holds).
//   3. Each candidate is checked for VRAM feasibility via evaluate().
//      The chosen layout is the LARGEST TP that fits: maximal sharding is
//      both the most memory-safe and the lowest-latency default in this
//      model. Every feasible candidate is returned as well so a UI can
//      offer the rest as one-click alternatives.

import type { Calibration } from './calibration';
import { IDEAL } from './calibration';
import type { GpuSpec, ModelSpec, ParallelLayout, QuantPrecision, Workload } from './types';
import type { PdMode } from './memory';
import { evaluate } from './metrics';

export interface SolverInput {
  model: ModelSpec;
  gpu: GpuSpec;
  gpusPerNode: number;
  numGpus: number;
  dp: number; // user-requested DP; sanitized to a divisor of numGpus
  ep: number; // user-requested EP; sanitized (MoE only, must divide replica)
  intraNodeBwGbps: number;
  interNodeBwGbps: number;
  workload: Workload;
  weightQuant: QuantPrecision;
  kvQuant: QuantPrecision;
  flashAttention: boolean;
  headroom: number;
  cal?: Calibration;
  // PD pool type hint: tells the solver to evaluate VRAM as if this layout
  // belongs to a prefill or decode pool in a PD-disaggregated deployment.
  // Absent = non-PD (colocated) mode.
  pdMode?: PdMode;
}

export interface SolverResult {
  // First feasible layout (largest TP), or null when nothing fits.
  chosen: ParallelLayout | null;
  // Largest-TP candidate regardless of feasibility; lets a UI render the
  // memory gap even when the model does not fit at all.
  bestEffort: ParallelLayout | null;
  // All feasible candidates, largest TP first.
  feasibleLayouts: ParallelLayout[];
  // Sanitized values actually used.
  dp: number;
  ep: number;
  // Human-readable notes about clamps/fallbacks (empty when input was clean).
  issues: string[];
}

// Largest divisor of n that is <= cap. n >= 1, cap >= 1.
function largestDivisorAtMost(n: number, cap: number): number {
  for (let d = Math.min(cap, n); d >= 1; d--) {
    if (n % d === 0) return d;
  }
  return 1;
}

export function solveParallelLayout(input: SolverInput): SolverResult {
  const issues: string[] = [];
  const cal = input.cal ?? IDEAL;

  // DP must divide the GPU count; fall back to DP = 1 otherwise.
  let dp = Math.max(1, Math.floor(input.dp) || 1);
  if (input.numGpus % dp !== 0) {
    issues.push(`DP=${dp} does not divide ${input.numGpus} GPUs; using DP=1`);
    dp = 1;
  }
  const perReplica = input.numGpus / dp;

  // EP is MoE-only and must divide the GPUs of one DP replica.
  let ep = Math.max(1, Math.floor(input.ep) || 1);
  if (ep > 1 && input.model.type !== 'moe') {
    issues.push(`EP=${ep} requires a MoE model ('${input.model.id}' is dense); using EP=1`);
    ep = 1;
  }
  if (ep > 1 && perReplica % ep !== 0) {
    const fixed = largestDivisorAtMost(perReplica, ep);
    issues.push(`EP=${ep} does not divide ${perReplica} GPUs per DP replica; using EP=${fixed}`);
    ep = fixed;
  }

  // Remainder is split as TP x PP. Candidates: divisors of `rest`, capped at
  // gpusPerNode, largest TP first.
  const rest = perReplica / ep;
  const candidates: ParallelLayout[] = [];
  for (let tp = Math.min(rest, input.gpusPerNode); tp >= 1; tp--) {
    if (rest % tp !== 0) continue;
    candidates.push({ tp, pp: rest / tp, ep, dp });
  }

  const feasibleLayouts: ParallelLayout[] = [];
  for (const layout of candidates) {
    const r = evaluate(
      {
        model: input.model,
        gpu: input.gpu,
        gpusPerNode: input.gpusPerNode,
        interNodeBwGbps: input.interNodeBwGbps,
        intraNodeBwGbps: input.intraNodeBwGbps,
        workload: input.workload,
        weightQuant: input.weightQuant,
        kvQuant: input.kvQuant,
        layout,
        flashAttention: input.flashAttention,
        headroom: input.headroom,
        // Pass PD mode hint so evaluate() uses pool-specific seqLen for VRAM.
        ...(input.pdMode ? { solverPdMode: input.pdMode } : {}),
      },
      cal,
    );
    if (r.ok && r.value.feasible) feasibleLayouts.push(layout);
  }

  return {
    chosen: feasibleLayouts[0] ?? null,
    bestEffort: candidates[0] ?? null,
    feasibleLayouts,
    dp,
    ep,
    issues,
  };
}
