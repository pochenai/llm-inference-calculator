// Throughput & latency vs batch size sweep (Recharts composed chart).

import { useMemo } from 'react';
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { evaluate } from '../../core/metrics';
import type { Calibration } from '../../core/calibration';
import type { SystemSpec } from '../../core/types';
import { deriveConstants } from '../../core/model';
import { vramBreakdown } from '../../core/memory';
import { fmtMs, fmtTps } from '../lib/format';
import { useI18n } from '../lib/i18n';

// Build x-axis batch values: powers of 2 up to the largest power of 2 <= maxBatch,
// then append maxBatch itself as the final point (skip if it's already a power of 2).
// Examples: maxBatch=78 -> [1,2,4,8,16,32,64,78]; maxBatch=128 -> [1,2,4,8,16,32,64,128].
function buildBatchSweep(maxBatch: number): number[] {
  const safeMax = Math.max(1, Math.floor(maxBatch));
  const powers: number[] = [];
  for (let p = 1; p <= safeMax; p *= 2) {
    powers.push(p);
  }
  // Append maxBatch only when it isn't already the last power of 2 in the list.
  const last = powers[powers.length - 1];
  if (last !== safeMax) {
    powers.push(safeMax);
  }
  return powers;
}

interface SweepPoint {
  batch: number;
  tps: number | null; // null = VRAM-infeasible at this batch
  e2eMs: number | null;
}

export function BatchSweepChart({ spec, cal }: { spec: SystemSpec; cal: Calibration }) {
  const { t } = useI18n();

  const points = useMemo<SweepPoint[]>(() => {
    // Compute VRAM-limited max batch (same logic as VramCard "System max").
    // When prefillRatio is set: steady-state bMax with ratio formula.
    // When prefillRatio is unset: bMaxFullLen (simple worst-case limit).
    let maxBatch: number;
    if (spec.disagg) {
      const pDp = spec.disagg.prefillLayout.dp;
      const dDp = spec.disagg.decodeLayout.dp;
      const derived = deriveConstants(spec.model, spec.weightQuant, spec.kvQuant);
      const memOpts = { flashAttention: spec.flashAttention, headroom: spec.headroom };
      const pMem = vramBreakdown(spec.model, derived, spec.gpu, spec.disagg.prefillLayout, spec.workload, { ...memOpts, pdMode: 'prefill' as const });
      const dMem = vramBreakdown(spec.model, derived, spec.gpu, spec.disagg.decodeLayout, spec.workload, { ...memOpts, pdMode: 'decode' as const });
      if (spec.workload.prefillRatio !== undefined) {
        const pRatio = spec.workload.prefillRatio;
        const dRatio = 1 - pRatio;
        const pBMax = pMem.bMax * pDp;
        const dBMax = dMem.bMax * dDp;
        maxBatch = Math.min(Math.floor(pBMax / pRatio), Math.floor(dBMax / dRatio));
      } else {
        const pBMax = pMem.bMaxFullLen * pDp;
        const dBMax = dMem.bMaxFullLen * dDp;
        maxBatch = Math.min(pBMax, dBMax);
      }
    } else {
      const derived = deriveConstants(spec.model, spec.weightQuant, spec.kvQuant);
      const mem = vramBreakdown(spec.model, derived, spec.gpu, spec.layout, spec.workload, {
        flashAttention: spec.flashAttention,
        headroom: spec.headroom,
      });
      const dp = spec.layout.dp;
      maxBatch = spec.workload.prefillRatio !== undefined
        ? mem.bMax * dp
        : mem.bMaxFullLen * dp;
    }
    const batches = buildBatchSweep(Math.max(1, maxBatch));
    return batches.map((b) => {
      const r = evaluate({ ...spec, workload: { ...spec.workload, batchSize: b } }, cal);
      if (!r.ok || !r.value.feasible) return { batch: b, tps: null, e2eMs: null };
      return { batch: b, tps: r.value.throughputTps, e2eMs: r.value.e2eMs };
    });
  }, [spec, cal]);

  if (points.every((p) => p.tps === null)) {
    return (
      <div className="card">
        <h3 className="card-title">{t('title.batch_sweep')}</h3>
        <div className="card-body">
          <div className="muted small">{t('note.no_feasible_batch')}</div>
        </div>
      </div>
    );
  }

  const infeasible = points.filter((p) => p.tps === null).map((p) => p.batch);

  return (
    <div className="card">
      <h3 className="card-title">{t('title.batch_sweep')}</h3>
      <div className="card-body">
        <div className="sweep-chart">
          <ResponsiveContainer width="100%" height={290}>
            <ComposedChart data={points} margin={{ top: 20, right: 12, left: 8, bottom: 24 }}>
              <CartesianGrid stroke="#eceef1" vertical={false} />
              <XAxis
                dataKey="batch"
                tick={{ fontSize: 11, fill: '#68707d' }}
                tickLine={false}
                axisLine={{ stroke: '#c9ced6' }}
                label={{
                  value: t('label.batch_size_axis'),
                  position: 'insideBottom',
                  offset: -16,
                  fill: '#68707d',
                  fontSize: 11,
                }}
              />
              <YAxis
                yAxisId="tps"
                width={52}
                tickFormatter={(v: number) => fmtTps(v)}
                tick={{ fontSize: 11, fill: '#68707d' }}
                tickLine={{ stroke: '#c9ced6' }}
                axisLine={{ stroke: '#c9ced6' }}
                label={{ value: t('label.throughput_axis'), position: 'top', offset: 8, fill: '#68707d', fontSize: 11 }}
              />
              {/* Right Y-axis label: absolute-positioned text to avoid recharts'
                  confusing "offset pushes toward centre" semantics. */}
              <YAxis
                yAxisId="lat"
                orientation="right"
                width={52}
                tickFormatter={(v: number) => fmtMs(v, { forceUnit: 's' })}
                tick={{ fontSize: 11, fill: '#68707d' }}
                tickLine={{ stroke: '#c9ced6' }}
                axisLine={{ stroke: '#c9ced6' }}
                label={{ value: t('label.latency_axis'), position: 'top', offset: 8, fill: '#68707d', fontSize: 11 }}
              />
              <Tooltip content={<SweepTooltip />} cursor={{ fill: 'rgba(59, 91, 219, 0.06)' }} />
              <Legend position="top" wrapperStyle={{ fontSize: 12 }} />
              <Bar
                yAxisId="tps"
                dataKey="tps"
                name={t('label.throughput_tps')}
                fill="rgba(59, 91, 219, 0.75)"
                radius={[3, 3, 0, 0]}
                maxBarSize={34}
              />
              <Line
                yAxisId="lat"
                dataKey="e2eMs"
                name={t('label.e2e_latency')}
                stroke="#b197fc"
                strokeWidth={1.8}
                dot={{ r: 2.5, fill: '#b197fc', strokeWidth: 0 }}
                connectNulls={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        {infeasible.length > 0 && (
          <div className="muted small">
            {t('note.oom_batch', { batches: infeasible.join(', ') })}
          </div>
        )}
      </div>
    </div>
  );
}

interface TooltipEntry {
  value?: number | string | null;
  dataKey?: string | number;
}

interface SweepTooltipProps {
  active?: boolean;
  payload?: TooltipEntry[];
  label?: string | number;
}

function SweepTooltip({ active, payload, label }: SweepTooltipProps) {
  const { t } = useI18n();

  if (!active || !payload || payload.length === 0) return null;
  const tps = payload.find((p) => p.dataKey === 'tps')?.value;
  const lat = payload.find((p) => p.dataKey === 'e2eMs')?.value;
  if (typeof tps !== 'number' && typeof lat !== 'number') {
    return <div className="sweep-tip">batch = {label}：{t('label.oom_tooltip').split(': ')[1] || 'VRAM overflow'}</div>;
  }
  return (
    <div className="sweep-tip">
      <div className="sweep-tip-head">batch = {label}</div>
      {typeof tps === 'number' && <div>{t('label.throughput_tooltip', { tps: fmtTps(tps) })}</div>}
      {typeof lat === 'number' && <div>{t('label.e2e_tooltip', { ms: fmtMs(lat) })}</div>}
    </div>
  );
}
