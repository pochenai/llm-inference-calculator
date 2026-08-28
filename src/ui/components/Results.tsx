// Right-hand results panel: status, headline metrics, layout, VRAM, phase details.

import { lazy, Suspense } from 'react';
import type { EvaluationResult } from '../../core/metrics';
import type { Result } from '../../core/errors';
import type { GpuSpec, ModelSpec, ParallelLayout, SystemSpec } from '../../core/types';
import type { SolverResult } from '../../core/solver';
import type { Calibration } from '../../core/calibration';
import { fmtBytes, fmtInt, fmtMs, fmtPct, fmtTps } from '../lib/format';
import { useI18n } from '../lib/i18n';

// Lazy-loaded so recharts does not land in the initial bundle.
const BatchSweepChart = lazy(async () => ({
  default: (await import('./BatchSweepChart')).BatchSweepChart,
}));

export interface ResultsProps {
  model: ModelSpec;
  gpu: GpuSpec;
  spec: SystemSpec;
  cal: Calibration;
  result: Result<EvaluationResult>;
  solved: SolverResult;
  effectiveLayout: ParallelLayout;
  layoutIsOverride: boolean;
  onPickLayout: (l: ParallelLayout | null) => void;
  disaggOn: boolean;
  prefillSolved: SolverResult | null;
  decodeSolved: SolverResult | null;
  prefillLayoutIsOverride: boolean;
  decodeLayoutIsOverride: boolean;
  onPickPrefillLayout: (l: ParallelLayout | null) => void;
  onPickDecodeLayout: (l: ParallelLayout | null) => void;
  warnings: string[];
}

function layoutLabel(l: ParallelLayout): string {
  return `TP ${l.tp} · EP ${l.ep} · PP ${l.pp} · DP ${l.dp}`;
}

function sameLayout(a: ParallelLayout, b: ParallelLayout): boolean {
  return a.tp === b.tp && a.pp === b.pp && a.ep === b.ep && a.dp === b.dp;
}

export function Results(props: ResultsProps) {
  const { result } = props;
  const { t } = useI18n();

  if (!result.ok) {
    return (
      <div className="card status-card err">
        <strong>{t('error.cannot_compute')}</strong>
        <span className="status-detail">
          [{result.error.code}] {result.error.message}
        </span>
      </div>
    );
  }
  const r = result.value;
  return (
    <>
      <StatusBanner {...props} r={r} />
      <MetricTiles r={r} />
      <LayoutCard {...props} />
      <VramCard {...props} r={r} />
      <PhaseCard r={r} model={props.model} spec={props.spec} disaggOn={props.disaggOn} />
      {r.speculative && <SpeculativeCard speculative={r.speculative} />}
      <Suspense
        fallback={
          <div className="card">
            <div className="muted small">{t('label.loading_chart')}</div>
          </div>
        }
      >
        <BatchSweepChart spec={props.spec} cal={props.cal} />
      </Suspense>
      <Warnings {...props} />
    </>
  );
}

function StatusBanner(props: ResultsProps & { r: EvaluationResult }) {
  const { r, model, gpu, spec } = props;
  const { t } = useI18n();

  if (!r.feasible) {
    const over = r.memory.totalBytes - r.memory.capacityBytes;
    return (
      <div className="card status-card err">
        <strong>{t('error.oom', { overBytes: fmtBytes(Math.max(0, over)) })}</strong>
        <span className="status-detail">
          {model.name} @ {spec.weightQuant.toUpperCase()} / {gpu.name} —{' '}
          {t('hint.oom')}
        </span>
      </div>
    );
  }
  const totalGpus = props.disaggOn
    ? (spec.disagg?.prefillGpus ?? 0) + (spec.disagg?.decodeGpus ?? 0)
    : spec.layout.tp * spec.layout.ep * spec.layout.pp * spec.layout.dp;
  return (
    <div className="card status-card ok">
      <strong>{t('status.deployable')}</strong>
      <span className="status-detail">
        {model.name}（{model.type === 'moe' ? 'MoE' : 'Dense'} · {model.paramsB}B） on {gpu.name} × {totalGpus}
        {props.disaggOn
          ? `（${spec.disagg?.prefillGpus ?? 0}P [batch ${r.prefillBatchSize}] + ${spec.disagg?.decodeGpus ?? 0}D [batch ${r.decodeBatchSize}]）`
          : ''}
      </span>
    </div>
  );
}

function MetricTiles({ r }: { r: EvaluationResult }) {
  const { t } = useI18n();
  const tiles = [
    { label: t('metric.ttft'), value: fmtMs(r.ttftMs) },
    { label: t('metric.tpot'), value: fmtMs(r.tpotMs) },
    { label: t('metric.e2e'), value: fmtMs(r.e2eMs) },
    { label: t('metric.throughput'), value: `${fmtTps(r.throughputTps)} tok/s` },
  ];
  return (
    <div className="metric-grid">
      {tiles.map((tile) => (
        <div className="card metric-tile" key={tile.label}>
          <span className="metric-value">{tile.value}</span>
          <span className="metric-label">{tile.label}</span>
        </div>
      ))}
    </div>
  );
}

function LayoutCard(props: ResultsProps) {
  const {
    solved,
    effectiveLayout,
    onPickLayout,
    disaggOn,
    prefillSolved,
    decodeSolved,
    onPickPrefillLayout,
    onPickDecodeLayout,
    spec,
  } = props;
  const { t } = useI18n();

  // Sort layouts by TP (desc) => EP (desc) => PP (desc), DP is fixed
  function sortLayouts(layouts: ParallelLayout[]): ParallelLayout[] {
    return [...layouts].sort((a, b) => {
      if (b.tp !== a.tp) return b.tp - a.tp;
      if (b.ep !== a.ep) return b.ep - a.ep;
      return b.pp - a.pp;
    });
  }

  if (disaggOn) {
    const prefillLayout = spec.disagg?.prefillLayout ?? { tp: 1, pp: 1, ep: 1, dp: 1 };
    const decodeLayout = spec.disagg?.decodeLayout ?? { tp: 1, pp: 1, ep: 1, dp: 1 };

    const prefillLayouts = prefillSolved ? sortLayouts(prefillSolved.feasibleLayouts) : [];
    const decodeLayouts = decodeSolved ? sortLayouts(decodeSolved.feasibleLayouts) : [];

    // Optimal PD GPU allocation info
    const optFrac = props.result.ok ? props.result.value.optimalPrefillFraction : undefined;
    const optGpus = props.result.ok ? props.result.value.optimalPrefillGpus : undefined;
    const pGpus = spec.disagg?.prefillGpus ?? 0;
    const dGpus = spec.disagg?.decodeGpus ?? 0;
    const totalPdGpus = pGpus + dGpus;

    return (
      <div className="card">
        <h3 className="card-title">{t('title.layout_pd')}</h3>
        <div className="card-body">
          <div className="disagg-layout">
            <div className="disagg-pool">
              <b>{t('label.prefill_pool')}</b>
              <span className="muted small">（batch = {props.result.ok ? props.result.value.prefillBatchSize : '–'}）</span>
              {prefillSolved?.chosen === null && (
                <div className="note err-note">{t('note.no_feasible_layout')}</div>
              )}
              {prefillLayouts.length > 0 ? (
                <div className="chip-row">
                  {prefillLayouts.map((l) => (
                    <button
                      key={layoutLabel(l)}
                      type="button"
                      className={`chip${sameLayout(l, prefillLayout) ? ' active' : ''}`}
                      onClick={() => onPickPrefillLayout(l)}
                    >
                      {layoutLabel(l)}
                    </button>
                  ))}
                </div>
              ) : (
                <div className="muted small">{t('note.no_layout')}</div>
              )}
            </div>

            <div className="disagg-pool">
              <b>{t('label.decode_pool')}</b>
              <span className="muted small">（batch = {props.result.ok ? props.result.value.decodeBatchSize : '–'}）</span>
              {decodeSolved?.chosen === null && (
                <div className="note err-note">{t('note.no_feasible_layout')}</div>
              )}
              {decodeLayouts.length > 0 ? (
                <div className="chip-row">
                  {decodeLayouts.map((l) => (
                    <button
                      key={layoutLabel(l)}
                      type="button"
                      className={`chip${sameLayout(l, decodeLayout) ? ' active' : ''}`}
                      onClick={() => onPickDecodeLayout(l)}
                    >
                      {layoutLabel(l)}
                    </button>
                  ))}
                </div>
              ) : (
                <div className="muted small">{t('note.no_layout')}</div>
              )}
            </div>
          </div>
          <div className="muted small">
            {t('note.layout_constraint', { gpusPerNode: spec.gpusPerNode })}
          </div>
          {optFrac !== undefined && optGpus !== undefined && (
            <div className="muted small" style={{ marginTop: 6 }}>
              <b>{t('label.optimal_gpu_alloc')}</b>：Prefill {optGpus} / Decode {totalPdGpus - optGpus}
              （Prefill {(optFrac * 100).toFixed(1)}%）
              {optGpus !== pGpus
                ? t('note.suggest_prefill', { optGpus })
                : t('note.optimal_current')}
              <br />
              {t('note.pipeline_balance')}
            </div>
          )}
          <SolverIssues issues={[...(prefillSolved?.issues ?? []), ...(decodeSolved?.issues ?? [])]} />
        </div>
      </div>
    );
  }

  // Sort layouts by TP (desc) => EP (desc) => PP (desc), DP is fixed
  const allLayouts = sortLayouts(solved.feasibleLayouts);

  return (
    <div className="card">
      <h3 className="card-title">{t('title.layout')}</h3>
      <div className="card-body">
        {solved.chosen === null && <div className="note err-note">{t('note.no_layout_fits')}</div>}
        {allLayouts.length > 0 ? (
          <div className="chip-row">
            {allLayouts.map((l) => (
              <button
                key={layoutLabel(l)}
                type="button"
                className={`chip${sameLayout(l, effectiveLayout) ? ' active' : ''}`}
                onClick={() => onPickLayout(l)}
              >
                {layoutLabel(l)}
              </button>
            ))}
          </div>
        ) : (
          <div className="muted small">{t('note.no_layout')}</div>
        )}
        <div className="muted small">
          {t('note.layout_constraint', { gpusPerNode: props.spec.gpusPerNode })}
        </div>
        <SolverIssues issues={solved.issues} />
      </div>
    </div>
  );
}

function SolverIssues({ issues }: { issues: string[] }) {
  if (issues.length === 0) return null;
  return (
    <ul className="issue-list">
      {issues.map((s) => (
        <li key={s}>{s}</li>
      ))}
    </ul>
  );
}

function VramBar({
  label,
  mem,
  dp,
  steadyState,
  systemBMax,
}: {
  label?: string;
  mem: EvaluationResult['memory'];
  dp?: number;
  // Whether to show steady-state bMax (true) or full-load bMaxFullLen (false).
  steadyState: boolean;
  // When this bar represents one pool of a PD pair, show the system-level
  // bMax that THIS pool limits the system to (bMax_pool / pool_fraction).
  systemBMax?: number;
}) {
  const { t } = useI18n();
  const cap = mem.capacityBytes;
  const segs = [
    { name: t('seg.weights'), bytes: mem.weightsBytes, cls: 'seg-w' },
    { name: t('seg.kv'), bytes: mem.kvBytes, cls: 'seg-kv' },
    { name: t('seg.activation'), bytes: mem.activationBytes, cls: 'seg-act' },
    { name: t('seg.overhead'), bytes: mem.overheadBytes, cls: 'seg-ov' },
    // Draft model segments (only present when SD enabled)
    ...(mem.draftWeightsBytes
      ? [{ name: t('seg.draft_weights'), bytes: mem.draftWeightsBytes, cls: 'seg-draft-w' }]
      : []),
    ...(mem.draftKvBytes ? [{ name: t('seg.draft_kv'), bytes: mem.draftKvBytes, cls: 'seg-draft-kv' }] : []),
  ];
  const total = mem.totalBytes;
  const scale = total > cap ? cap / total : 1;
  const d = dp ?? 1;
  const poolBMax = steadyState ? mem.bMax * d : mem.bMaxFullLen * d;
  return (
    <div className="vram-block">
      {label ? <div className="vram-label">{label}</div> : null}
      <div className={`vram-bar${mem.feasible ? '' : ' over'}`}>
        {segs.map((s) => (
          <div
            key={s.name}
            className={`vram-seg ${s.cls}`}
            style={{ width: `${Math.max(0, (s.bytes / cap) * 100 * scale)}%` }}
            title={`${s.name}: ${fmtBytes(s.bytes)}`}
          />
        ))}
      </div>
      <div className="vram-meta">
        <span>
          {t('label.vram_usage', { total: fmtBytes(total), cap: fmtBytes(cap), pct: fmtPct(total / cap) })}
          {mem.feasible ? '' : t('label.over_capacity')}
        </span>
        <span>
          {t('label.max_batch_full', { poolBMax: fmtInt(poolBMax) })}
          {systemBMax !== undefined && (
            <>
              {t('label.system_max_batch', { systemBMax: fmtInt(systemBMax) })}
            </>
          )}
        </span>
      </div>
      <div className="vram-legend">
        {segs.map((s) => (
          <span key={s.name}>
            <i className={`dot ${s.cls}`} />
            {s.name} {fmtBytes(s.bytes)}
          </span>
        ))}
      </div>
    </div>
  );
}

function VramCard(props: ResultsProps & { r: EvaluationResult }) {
  const { r, disaggOn, spec } = props;
  const { t } = useI18n();

  // Whether steady-state mode is active (prefillRatio is set).
  const steadyState = spec.workload.prefillRatio !== undefined;

  // PD: compute system-level bMax from each pool's limit.
  // Steady-state: min(bMax_prefill / r, bMax_decode / (1-r))
  // Non-steady: min(bMaxFullLen_prefill, bMaxFullLen_decode)
  let pSystemBMax: number | undefined;
  let dSystemBMax: number | undefined;
  let totalSystemBMax: number | undefined;
  if (disaggOn && spec.disagg) {
    const pDp = spec.disagg.prefillLayout.dp;
    const dDp = spec.disagg.decodeLayout.dp;
    if (steadyState) {
      const pRatio = spec.workload.prefillRatio!;
      const dRatio = 1 - pRatio;
      const pBMax = r.memoryPrefillPool!.bMax * pDp;
      const dBMax = r.memory.bMax * dDp;
      pSystemBMax = Math.floor(pBMax / pRatio);
      dSystemBMax = Math.floor(dBMax / dRatio);
    } else {
      pSystemBMax = r.memoryPrefillPool!.bMaxFullLen * pDp;
      dSystemBMax = r.memory.bMaxFullLen * dDp;
    }
    totalSystemBMax = Math.min(pSystemBMax, dSystemBMax);
  }
  return (
    <div className="card">
      <h3 className="card-title">{t('title.vram')}</h3>
      <div className="card-body">
        {disaggOn && r.memoryPrefillPool && spec.disagg ? (
          <>
            <VramBar
              label={t('label.prefill_pool_batch', { batch: r.prefillBatchSize })}
              mem={r.memoryPrefillPool}
              dp={spec.disagg.prefillLayout.dp}
              steadyState={steadyState}
              {...(pSystemBMax !== undefined ? { systemBMax: pSystemBMax } : {})}
            />
            <VramBar
              label={t('label.decode_pool_batch', { batch: r.decodeBatchSize })}
              mem={r.memory}
              dp={spec.disagg.decodeLayout.dp}
              steadyState={steadyState}
              {...(dSystemBMax !== undefined ? { systemBMax: dSystemBMax } : {})}
            />
            <div className="muted small">
              <b>{t('label.system_total_max_batch', { totalSystemBMax: fmtInt(totalSystemBMax!) })}</b>
            </div>
          </>
        ) : (
          <VramBar mem={r.memory} dp={spec.layout.dp} steadyState={steadyState} />
        )}
        <div className="muted small">
          {t('note.vram_capacity')}
        </div>
      </div>
    </div>
  );
}

function PhaseCard({
  r,
  model,
  spec,
  disaggOn,
}: {
  r: EvaluationResult;
  model: ModelSpec;
  spec: SystemSpec;
  disaggOn: boolean;
}) {
  const { t } = useI18n();
  const nIn = spec.workload.inputLen;
  const Bp = r.prefillBatchSize;
  const Bd = r.decodeBatchSize;
  // Prefill throughput: input tokens processed per second (B_p * N_in / TTFT).
  const prefillTps = r.ttftMs > 0 ? (Bp * nIn) / (r.ttftMs / 1e3) : 0;
  // Decode throughput: generated tokens per second across the batch (B_d / TPOT).
  const decodeTps = r.tpotMs > 0 ? Bd / (r.tpotMs / 1e3) : 0;
  return (
    <div className="card">
      <h3 className="card-title">{t('title.phase')}</h3>
      <div className="card-body">
        <div className="phase-grid">
        <table className="detail-table">
          <thead>
            <tr>
              <th colSpan={2}>
                {t('phase.prefill')}
                {disaggOn && <span className="muted small"> — batch {Bp}</span>}
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>{t('label.total_flops')}</td>
              <td>{(r.prefill.flops / 1e12).toFixed(1)} TFLOP</td>
            </tr>
            <tr>
              <td>{t('label.compute_time')}</td>
              <td>{fmtMs(r.prefill.tComputeMs)}</td>
            </tr>
            <tr>
              <td>{t('label.comm_time_total')}</td>
              <td>{fmtMs(r.prefill.tCommMs)}</td>
            </tr>
            <tr>
              <td>TTFT</td>
              <td>
                <b>{fmtMs(r.prefill.ttftMs)}</b>
              </td>
            </tr>
            <tr>
              <td>{t('label.throughput_input')}</td>
              <td>
                <b>{fmtTps(prefillTps)}</b>
              </td>
            </tr>
            {disaggOn && r.kvTransferExposedMs > 0 && (
              <tr>
                <td>{t('label.kv_transfer_exposed')}</td>
                <td>{fmtMs(r.kvTransferExposedMs)}</td>
              </tr>
            )}
            <tr>
              <td>{t('label.compute_util')}</td>
              <td>
                <b>{fmtPct(r.prefillComputeUtilization)}</b>
                <div className="muted small">
                  {r.prefillActualFlops.toFixed(1)} / {r.prefillPeakFlops.toFixed(1)} TFLOPS
                </div>
              </td>
            </tr>
          </tbody>
        </table>
        <table className="detail-table">
          <thead>
            <tr>
              <th colSpan={2}>
                {t('phase.decode')}
                {disaggOn && <span className="muted small"> — batch {Bd}</span>}
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>{t('label.weight_read')}</td>
              <td>{fmtBytes(r.decode.weightsReadBytes)}</td>
            </tr>
            <tr>
              <td>{t('label.kv_read')}</td>
              <td>{fmtBytes(r.decode.kvReadBytes)}</td>
            </tr>
            <tr>
              <td>{t('label.bandwidth_time')}</td>
              <td>{fmtMs(r.decode.tBandwidthMs)}</td>
            </tr>
            <tr>
              <td>{t('label.comm_time_total')}</td>
              <td>{fmtMs(r.decode.tCommMs)}</td>
            </tr>
            {model.type === 'moe' && (
              <tr>
                <td>{t('label.expert_coverage')}</td>
                <td>{fmtPct(r.decode.expertCoverage)}</td>
              </tr>
            )}
            <tr>
              <td>TPOT</td>
              <td>
                <b>{fmtMs(r.decode.tpotMs)}</b>
              </td>
            </tr>
            <tr>
              <td>{t('label.throughput_output')}</td>
              <td>
                <b>{fmtTps(decodeTps)}</b>
              </td>
            </tr>
            <tr>
              <td>{t('label.bandwidth_util')}</td>
              <td>
                <b>{fmtPct(r.decodeBandwidthUtilization)}</b>
                <div className="muted small">
                  {r.decodeActualBandwidth.toFixed(0)} / {r.decodePeakBandwidth.toFixed(0)} GB/s
                </div>
              </td>
            </tr>
          </tbody>
        </table>
        </div>
        <div className="muted small">
          {t('note.comm_overlap')}
          {disaggOn && (
            <>
              <br />
              {t('note.steady_state_split', { Bp, Bd, total: Bp + Bd })}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function Warnings(props: ResultsProps) {
  if (props.warnings.length === 0) return null;
  return (
    <div className="card warn-card">
      {props.warnings.map((w) => (
        <div key={w} className="warn-line">
          ⚠ {w}
        </div>
      ))}
    </div>
  );
}

function SpeculativeCard({ speculative }: { speculative: NonNullable<EvaluationResult['speculative']> }) {
  const { t } = useI18n();
  return (
    <div className="card">
      <h3 className="card-title">{t('title.speculative')}</h3>
      <div className="card-body">
        <div className="phase-grid">
          <table className="detail-table">
            <thead>
              <tr>
                <th colSpan={2}>{t('label.draft_model_info')}</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>{t('label.model_name')}</td>
                <td>{speculative.draftModelName}</td>
              </tr>
              <tr>
                <td>{t('label.gamma')}</td>
                <td>{speculative.gamma}</td>
              </tr>
              <tr>
                <td>{t('label.acceptance_rate')}</td>
                <td>{fmtPct(speculative.acceptanceRate)}</td>
              </tr>
              <tr>
                <td>{t('label.expected_tokens')}</td>
                <td>{speculative.expectedTokensPerCycle.toFixed(2)}</td>
              </tr>
            </tbody>
          </table>
          <table className="detail-table">
            <thead>
              <tr>
                <th colSpan={2}>{t('label.timing_analysis')}</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>{t('label.draft_step_time')}</td>
                <td>{fmtMs(speculative.draftStepMs)}</td>
              </tr>
              <tr>
                <td>{t('label.verify_step_time')}</td>
                <td>{fmtMs(speculative.verifyStepMs)}</td>
              </tr>
              <tr>
                <td>{t('label.cycle_time')}</td>
                <td>{fmtMs(speculative.cycleTimeMs)}</td>
              </tr>
              <tr>
                <td>{t('label.tpot_speculative')}</td>
                <td>
                  <b>{fmtMs(speculative.verifyStepMs / speculative.expectedTokensPerCycle + speculative.draftStepMs * speculative.gamma / speculative.expectedTokensPerCycle)}</b>
                </td>
              </tr>
              <tr>
                <td>{t('label.tpot_baseline')}</td>
                <td>{fmtMs(speculative.baselineTpotMs)}</td>
              </tr>
              <tr>
                <td>{t('label.speedup')}</td>
                <td>
                  <b>{speculative.speedup.toFixed(2)}×</b>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="muted small">
          {t('note.speculative')}
        </div>
      </div>
    </div>
  );
}
