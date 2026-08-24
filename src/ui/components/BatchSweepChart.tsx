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
import { fmtMs, fmtTps } from '../lib/format';

const SWEEP_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256];

interface SweepPoint {
  batch: number;
  tps: number | null; // null = VRAM-infeasible at this batch
  e2eMs: number | null;
}

export function BatchSweepChart({ spec, cal }: { spec: SystemSpec; cal: Calibration }) {
  const points = useMemo<SweepPoint[]>(() => {
    // Always include the currently selected batch so it shows on the chart.
    const batches = [...SWEEP_BATCHES];
    const current = spec.workload.batchSize;
    if (!batches.includes(current)) {
      batches.push(current);
      batches.sort((a, b) => a - b);
    }
    return batches.map((b) => {
      const r = evaluate({ ...spec, workload: { ...spec.workload, batchSize: b } }, cal);
      if (!r.ok || !r.value.feasible) return { batch: b, tps: null, e2eMs: null };
      return { batch: b, tps: r.value.throughputTps, e2eMs: r.value.e2eMs };
    });
  }, [spec, cal]);

  if (points.every((p) => p.tps === null)) {
    return (
      <div className="card">
        <h3 className="card-title">Throughput & Latency vs Batch Size</h3>
        <div className="card-body">
          <div className="muted small">当前配置下没有可行的 batch。</div>
        </div>
      </div>
    );
  }

  const infeasible = points.filter((p) => p.tps === null).map((p) => p.batch);

  return (
    <div className="card">
      <h3 className="card-title">Throughput & Latency vs Batch Size</h3>
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
                  value: 'Batch Size',
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
                label={{ value: 'Throughput', position: 'top', offset: 8, fill: '#68707d', fontSize: 11 }}
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
                label={{ value: 'Latency', position: 'top', offset: 8, fill: '#68707d', fontSize: 11 }}
              />
              <Tooltip content={<SweepTooltip />} cursor={{ fill: 'rgba(59, 91, 219, 0.06)' }} />
              <Legend position="top" wrapperStyle={{ fontSize: 12 }} />
              <Bar
                yAxisId="tps"
                dataKey="tps"
                name="吞吐 (tok/s)"
                fill="rgba(59, 91, 219, 0.75)"
                radius={[3, 3, 0, 0]}
                maxBarSize={34}
              />
              <Line
                yAxisId="lat"
                dataKey="e2eMs"
                name="端到端延迟"
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
            batch = {infeasible.join(', ')} 显存不足（超过最大 Batch size），未在图中显示。
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
  if (!active || !payload || payload.length === 0) return null;
  const tps = payload.find((p) => p.dataKey === 'tps')?.value;
  const lat = payload.find((p) => p.dataKey === 'e2eMs')?.value;
  if (typeof tps !== 'number' && typeof lat !== 'number') {
    return <div className="sweep-tip">batch = {label}：显存不足</div>;
  }
  return (
    <div className="sweep-tip">
      <div className="sweep-tip-head">batch = {label}</div>
      {typeof tps === 'number' && <div>吞吐：{fmtTps(tps)} tok/s</div>}
      {typeof lat === 'number' && <div>端到端延迟：{fmtMs(lat)}</div>}
    </div>
  );
}
