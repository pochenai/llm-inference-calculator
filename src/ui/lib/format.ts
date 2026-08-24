// Formatting helpers for the UI (display only, never used by the core).

export function fmtBytes(bytes: number): string {
  if (!Number.isFinite(bytes)) return '–';
  const gb = bytes / 1e9;
  if (Math.abs(gb) >= 1000) return `${(gb / 1000).toFixed(2)} TB`;
  if (Math.abs(gb) >= 100) return `${gb.toFixed(0)} GB`;
  if (Math.abs(gb) >= 1) return `${gb.toFixed(1)} GB`;
  const mb = bytes / 1e6;
  if (Math.abs(mb) >= 1) return `${mb.toFixed(1)} MB`;
  return `${(bytes / 1e3).toFixed(1)} KB`;
}

export function fmtMs(ms: number, opts?: { forceUnit?: string }): string {
  if (!Number.isFinite(ms)) return '–';
  // If a unit is forced, always output that unit.
  if (opts?.forceUnit === 's') {
    const sec = ms / 1000;
    if (sec < 0.001) return `${(sec * 1e6).toFixed(0)}µs`;
    if (sec < 1) return `${sec.toFixed(2)}s`;
    if (sec < 60) return `${sec.toFixed(2)}s`;
    return `${(sec / 60).toFixed(1)}min`;
  }
  if (ms < 1) return `${(ms * 1000).toFixed(0)} µs`;
  if (ms < 1000) return `${ms.toFixed(ms < 10 ? 2 : 1)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(2)} s`;
  return `${(ms / 60_000).toFixed(1)} min`;
}

export function fmtTps(tps: number): string {
  if (!Number.isFinite(tps)) return '–';
  if (tps >= 10_000) return `${(tps / 1000).toFixed(1)}k`;
  if (tps >= 100) return tps.toFixed(0);
  return tps.toFixed(1);
}

export function fmtInt(n: number): string {
  if (!Number.isFinite(n)) return '–';
  return Math.round(n).toLocaleString('en-US');
}

export function fmtCtx(tokens: number): string {
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M`;
  if (tokens >= 1000) return `${(tokens / 1000).toFixed(0)}k`;
  return String(tokens);
}

export function fmtPct(frac: number): string {
  if (!Number.isFinite(frac)) return '–';
  return `${(frac * 100).toFixed(0)}%`;
}
