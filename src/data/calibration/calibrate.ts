// Formal benchmark comparison: ideal-value predictions vs measured numbers.
// English comments.
//
// For every LAT entry with a known model+gpu spec we run evaluate() and compare
// TTFT / TPOT / throughput / E2E against the measured values; metrics the source
// did not report render as "-". THR entries are compared against the decode
// roofline ceiling BW_total / (kv_per_token * S_avg) as an upper bound.

import { IDEAL, DEFAULT_ALPHA_INTRA_MS, DEFAULT_ALPHA_INTER_MS } from '../../core/calibration';
import type { Calibration } from '../../core/calibration';
import { evaluate } from '../../core/metrics';
import { deriveConstants } from '../../core/model';
import { MEASUREMENTS } from './measurements';
import type { Measurement } from './measurements';
import { model } from './models';
import { gpu } from './gpus';

// Small-context latency entries: batch=1 collectives cannot be overlapped, so
// expose TP/EP comm and use the default per-call alpha.
const CAL_LAT_SMALL: Calibration = {
  ...IDEAL,
  alphaIntraMs: DEFAULT_ALPHA_INTRA_MS,
  alphaInterMs: DEFAULT_ALPHA_INTER_MS,
  tpCommOverlap: 0,
  epCommOverlap: 0,
};

// Long-context prefill-dominated entries follow the README comparison basis
// (pure compute ideal; PP/TP comm treated as hidden).
function calibrationFor(e: Measurement): Calibration {
  return e.inputLen > 8192 ? IDEAL : CAL_LAT_SMALL;
}

function fmt(x: number | undefined, digits = 1): string {
  if (x === undefined || !Number.isFinite(x)) return '-';
  return x >= 1000 ? Math.round(x).toLocaleString('en-US') : x.toFixed(digits);
}

function ratioCell(measured: number | undefined, ideal: number | undefined): string {
  if (measured === undefined || ideal === undefined || ideal === 0) return '-';
  return `${(measured / ideal).toFixed(2)}x`;
}

interface IdealMetrics {
  ttftMs?: number;
  tpotMs?: number;
  throughputTps?: number;
  e2eMs?: number;
}

function idealFor(e: Measurement): IdealMetrics {
  if (e.protocol === 'THR') {
    // Decode roofline ceiling at the VRAM-limited max batch, so weight reads are
    // amortized correctly (matters for large models where B cannot grow freely).
    const g = gpu(e.gpuId);
    const derived = deriveConstants(model(e.modelId), e.weightQuant, e.kvQuant);
    const sAvg = e.inputLen + e.outputLen / 2;
    const wPerGpu = derived.wBytesTotal / e.gpuCount;
    const kvPerTokPerGpu = derived.kv.bytesPerToken / e.gpuCount;
    const budget = Math.max(0, g.vramGb * 1e9 * 0.9 - wPerGpu);
    const bMax = Math.max(1, Math.floor(budget / (kvPerTokPerGpu * sAvg)));
    const bwAgg = g.bwGbps * 1e9 * e.gpuCount;
    const ceiling = (bMax * bwAgg) / (derived.wBytesTotal + derived.kv.bytesPerToken * sAvg * bMax);
    return { throughputTps: ceiling };
  }

  const cal = calibrationFor(e);
  const r = evaluate(
    {
      model: model(e.modelId),
      gpu: gpu(e.gpuId),
      gpusPerNode: e.gpusPerNode,
      interNodeBwGbps: 400,
      workload: { batchSize: e.batch, inputLen: e.inputLen, outputLen: e.outputLen },
      weightQuant: e.weightQuant,
      kvQuant: e.kvQuant,
      layout: e.layout,
      flashAttention: true,
      headroom: 0.1,
    },
    cal,
  );
  if (!r.ok) return {};

  // Prefill-only entries (outputLen <= 1): throughput means input tokens / TTFT.
  if (e.outputLen <= 1) {
    return { ttftMs: r.value.ttftMs, throughputTps: e.inputLen / (r.value.ttftMs / 1e3) };
  }
  return {
    ttftMs: r.value.ttftMs,
    tpotMs: r.value.tpotMs,
    throughputTps: r.value.throughputTps,
    e2eMs: r.value.e2eMs,
  };
}

// Render the full comparison as a fixed-width table: every cell is padded to its
// column width, so columns stay aligned in a terminal (still valid markdown).
export function renderCalibration(): string {
  const header = ['id', 'proto', 'setup', 'TTFT m/i (r)', 'TPOT m/i (r)', 'Thr m/i (r)', 'E2E m/i (r)'];
  const rows: string[][] = [header];
  for (const e of MEASUREMENTS) {
    const ideal = idealFor(e);
    const setup = `${e.modelId} @ ${e.gpuCount}x${e.gpuId} TP${e.layout.tp}PP${e.layout.pp} B${e.batch} ${e.inputLen}/${e.outputLen} ${e.weightQuant}`;
    const ttft = `${fmt(e.measured.ttftMs, 0)} / ${fmt(ideal.ttftMs, 0)} (${ratioCell(e.measured.ttftMs, ideal.ttftMs)})`;
    const tpot = `${fmt(e.measured.tpotMs)} / ${fmt(ideal.tpotMs)} (${ratioCell(e.measured.tpotMs, ideal.tpotMs)})`;
    const thr = `${fmt(e.measured.throughputTps, 0)} / ${fmt(ideal.throughputTps, 0)} (${ratioCell(e.measured.throughputTps, ideal.throughputTps)})`;
    const e2e = `${fmt(e.measured.e2eMs, 0)} / ${fmt(ideal.e2eMs, 0)} (${ratioCell(e.measured.e2eMs, ideal.e2eMs)})`;
    rows.push([e.id, e.protocol, setup, ttft, tpot, thr, e2e]);
  }

  const widths = header.map((_, c) => Math.max(...rows.map((r) => (r[c] ?? '').length)));
  const line = (r: string[]): string =>
    '| ' + r.map((cell, c) => cell.padEnd(widths[c] ?? 0)).join(' | ') + ' |';
  const lines = rows.map(line);
  lines.splice(1, 0, '|' + widths.map((w) => '-'.repeat(w + 2)).join('|') + '|');

  lines.push('');
  lines.push('m = measured, i = ideal, r = measured/ideal ratio. "-" = not reported / not applicable.');
  lines.push('LAT ratios ~2-5x mark the calibration region (MFU/alpha); THR ratio <1 = fraction of the decode roofline ceiling.');
  return lines.join('\n');
}
