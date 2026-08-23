// Global error type + Result contract for the calculator core.
//
// Design goal: the public entry point evaluate() NEVER throws. It returns a
// Result<EvaluationResult>, so a UI can branch on r.ok and render r.error
// without wrapping calls in try/catch. Internal helpers still throw CalcError
// (typed, coded); evaluate() catches them at the boundary.

export type CalcErrorCode =
  | 'contradictory-layers' // special-layer counts exceed total layers
  | 'inconsistent-moe' // paramsB/activeParamsB/experts/expertsPerToken don't line up
  | 'negative-seqlen' // negative sequence length passed by the caller
  | 'invalid-layout' // TP*EP*PP*DP != N_gpu, EP on dense, cross-node TP, ...
  | 'missing-peak-flops' // GPU datasheet lacks the requested precision column
  | 'internal'; // unexpected error wrapped at the evaluate() boundary

export class CalcError extends Error {
  readonly code: CalcErrorCode;

  constructor(code: CalcErrorCode, message: string) {
    super(message);
    this.name = 'CalcError';
    this.code = code;
  }
}

export type Result<T> = { ok: true; value: T } | { ok: false; error: CalcError };

export function ok<T>(value: T): Result<T> {
  return { ok: true, value };
}

export function err<T>(error: CalcError): Result<T> {
  return { ok: false, error };
}

// Narrowing helper for call sites that expect success (tests, CLI demos).
// Throws if the result is an error.
export function unwrap<T>(result: Result<T>): T {
  if (!result.ok) {
    throw new Error(`CalcError [${result.error.code}]: ${result.error.message}`);
  }
  return result.value;
}
