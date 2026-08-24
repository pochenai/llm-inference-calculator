// Right-hand results panel: status, headline metrics, layout, VRAM, phase details.

import { lazy, Suspense } from 'react';
import type { EvaluationResult } from '../../core/metrics';
import type { Result } from '../../core/errors';
import type { GpuSpec, ModelSpec, ParallelLayout, SystemSpec } from '../../core/types';
import type { SolverResult } from '../../core/solver';
import type { Calibration } from '../../core/calibration';
import { fmtBytes, fmtInt, fmtMs, fmtPct, fmtTps } from '../lib/format';

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
  if (!result.ok) {
    return (
      <div className="card status-card err">
        <strong>无法计算</strong>
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
            <div className="muted small">加载图表…</div>
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
  if (!r.feasible) {
    const over = r.memory.totalBytes - r.memory.capacityBytes;
    return (
      <div className="card status-card err">
        <strong>显存不足（单卡超出 {fmtBytes(Math.max(0, over))}）</strong>
        <span className="status-detail">
          {model.name} @ {spec.weightQuant.toUpperCase()} / {gpu.name} —
          试试更低量化、更多 GPU、更高 EP/PP，或减小 batch / 序列长度。
        </span>
      </div>
    );
  }
  const totalGpus = props.disaggOn
    ? (spec.disagg?.prefillGpus ?? 0) + (spec.disagg?.decodeGpus ?? 0)
    : spec.layout.tp * spec.layout.ep * spec.layout.pp * spec.layout.dp;
  return (
    <div className="card status-card ok">
      <strong>可部署</strong>
      <span className="status-detail">
        {model.name}（{model.type === 'moe' ? 'MoE' : 'Dense'} · {model.paramsB}B） on {gpu.name} × {totalGpus}
        {props.disaggOn
          ? `（${spec.disagg?.prefillGpus ?? 0}P + ${spec.disagg?.decodeGpus ?? 0}D）`
          : ''}
      </span>
    </div>
  );
}

function MetricTiles({ r }: { r: EvaluationResult }) {
  const tiles = [
    { label: 'TTFT（首 token）', value: fmtMs(r.ttftMs) },
    { label: 'TPOT（每 token）', value: fmtMs(r.tpotMs) },
    { label: '端到端延迟', value: fmtMs(r.e2eMs) },
    { label: '系统吞吐', value: `${fmtTps(r.throughputTps)} tok/s` },
  ];
  return (
    <div className="metric-grid">
      {tiles.map((t) => (
        <div className="card metric-tile" key={t.label}>
          <span className="metric-value">{t.value}</span>
          <span className="metric-label">{t.label}</span>
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

    return (
      <div className="card">
        <h3 className="card-title">并行布局（PD 分离）</h3>
        <div className="card-body">
          <div className="disagg-layout">
            <div className="disagg-pool">
              <b>Prefill 池</b>
              {prefillSolved?.chosen === null && (
                <div className="note err-note">没有可行布局；以下为最接近的参照。</div>
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
                <div className="muted small">无可用布局。</div>
              )}
            </div>

            <div className="disagg-pool">
              <b>Decode 池</b>
              {decodeSolved?.chosen === null && (
                <div className="note err-note">没有可行布局；以下为最接近的参照。</div>
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
                <div className="muted small">无可用布局。</div>
              )}
            </div>
          </div>
          <div className="muted small">
            候选布局受「TP ≤ 每节点 GPU 数（{spec.gpusPerNode}）」约束，按 TP ⇒ EP ⇒ PP 排序
          </div>
          <SolverIssues issues={[...(prefillSolved?.issues ?? []), ...(decodeSolved?.issues ?? [])]} />
        </div>
      </div>
    );
  }

  // Sort layouts by TP (desc) => EP (desc) => PP (desc), DP is fixed
  const allLayouts = sortLayouts(solved.feasibleLayouts);

  return (
    <div className="card">
      <h3 className="card-title">并行布局</h3>
      <div className="card-body">
        {solved.chosen === null && <div className="note err-note">没有任何布局能装下当前配置；以下为最接近的参照。</div>}
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
          <div className="muted small">无可用布局。</div>
        )}
        <div className="muted small">
          候选布局受「TP ≤ 每节点 GPU 数（{props.spec.gpusPerNode}）」约束，按 TP ⇒ EP ⇒ PP 排序
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

function VramBar({ label, mem }: { label?: string; mem: EvaluationResult['memory'] }) {
  const cap = mem.capacityBytes;
  const segs = [
    { name: '权重', bytes: mem.weightsBytes, cls: 'seg-w' },
    { name: 'KV cache', bytes: mem.kvBytes, cls: 'seg-kv' },
    { name: '激活', bytes: mem.activationBytes, cls: 'seg-act' },
    { name: '开销', bytes: mem.overheadBytes, cls: 'seg-ov' },
    // Draft model segments (only present when SD enabled)
    ...(mem.draftWeightsBytes
      ? [{ name: 'Draft 权重', bytes: mem.draftWeightsBytes, cls: 'seg-draft-w' }]
      : []),
    ...(mem.draftKvBytes ? [{ name: 'Draft KV', bytes: mem.draftKvBytes, cls: 'seg-draft-kv' }] : []),
  ];
  const total = mem.totalBytes;
  const scale = total > cap ? cap / total : 1;
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
          {fmtBytes(total)} / {fmtBytes(cap)}（{fmtPct(total / cap)}）
          {mem.feasible ? '' : ' — 超出'}
        </span>
        <span>最大 Batch size = {fmtInt(mem.bMax)}</span>
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
  const { r, disaggOn } = props;
  return (
    <div className="card">
      <h3 className="card-title">显存占用（每卡）</h3>
      <div className="card-body">
        {disaggOn && r.memoryPrefillPool ? (
          <>
            <VramBar label="Prefill 池" mem={r.memoryPrefillPool} />
            <VramBar label="Decode 池" mem={r.memory} />
          </>
        ) : (
          <VramBar mem={r.memory} />
        )}
        <div className="muted small">
          容量已扣除 headroom 预留；最大 batch 为显存约束下可反推的上限。
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
  const B = spec.workload.batchSize;
  const nIn = spec.workload.inputLen;
  // Prefill throughput: input tokens processed per second (B * N_in / TTFT).
  const prefillTps = r.ttftMs > 0 ? (B * nIn) / (r.ttftMs / 1e3) : 0;
  // Decode throughput: generated tokens per second across the batch (B / TPOT).
  const decodeTps = r.tpotMs > 0 ? B / (r.tpotMs / 1e3) : 0;
  return (
    <div className="card">
      <h3 className="card-title">阶段明细</h3>
      <div className="card-body">
        <div className="phase-grid">
        <table className="detail-table">
          <thead>
            <tr>
              <th colSpan={2}>Prefill（计算受限）</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>总 FLOPs</td>
              <td>{(r.prefill.flops / 1e12).toFixed(1)} TFLOP</td>
            </tr>
            <tr>
              <td>计算时间</td>
              <td>{fmtMs(r.prefill.tComputeMs)}</td>
            </tr>
            <tr>
              <td>通信时间（总）</td>
              <td>{fmtMs(r.prefill.tCommMs)}</td>
            </tr>
            <tr>
              <td>TTFT</td>
              <td>
                <b>{fmtMs(r.prefill.ttftMs)}</b>
              </td>
            </tr>
            <tr>
              <td>吞吐（输入 token/s）</td>
              <td>
                <b>{fmtTps(prefillTps)}</b>
              </td>
            </tr>
            {disaggOn && r.kvTransferExposedMs > 0 && (
              <tr>
                <td>KV 传输（暴露）</td>
                <td>{fmtMs(r.kvTransferExposedMs)}</td>
              </tr>
            )}
            <tr>
              <td>算力利用率</td>
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
              <th colSpan={2}>Decode（带宽受限）</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>权重读取 / 卡·步</td>
              <td>{fmtBytes(r.decode.weightsReadBytes)}</td>
            </tr>
            <tr>
              <td>KV 读取 / 卡·步</td>
              <td>{fmtBytes(r.decode.kvReadBytes)}</td>
            </tr>
            <tr>
              <td>带宽时间</td>
              <td>{fmtMs(r.decode.tBandwidthMs)}</td>
            </tr>
            <tr>
              <td>通信时间（总）</td>
              <td>{fmtMs(r.decode.tCommMs)}</td>
            </tr>
            {model.type === 'moe' && (
              <tr>
                <td>专家覆盖率</td>
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
              <td>吞吐（输出 token/s）</td>
              <td>
                <b>{fmtTps(decodeTps)}</b>
              </td>
            </tr>
            <tr>
              <td>带宽利用率</td>
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
          注：「通信时间（总）」为原始通信量，进入延迟的是其暴露部分 = 总量 ×（1 −
          重叠系数）。重叠系数 = 1（理想模式）时通信被计算完全隐藏，TTFT / TPOT
          不含通信；可在「校准参数」面板调低 TP / EP / PP 通信重叠查看暴露的通信代价。
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
  return (
    <div className="card">
      <h3 className="card-title">投机采样明细</h3>
      <div className="card-body">
        <div className="phase-grid">
          <table className="detail-table">
            <thead>
              <tr>
                <th colSpan={2}>草稿模型</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>模型</td>
                <td>{speculative.draftModelName}</td>
              </tr>
              <tr>
                <td>γ（草稿步数）</td>
                <td>{speculative.gamma}</td>
              </tr>
              <tr>
                <td>接受率</td>
                <td>{fmtPct(speculative.acceptanceRate)}</td>
              </tr>
              <tr>
                <td>每周期期望 token</td>
                <td>{speculative.expectedTokensPerCycle.toFixed(2)}</td>
              </tr>
            </tbody>
          </table>
          <table className="detail-table">
            <thead>
              <tr>
                <th colSpan={2}>时序分析</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>草稿步时间</td>
                <td>{fmtMs(speculative.draftStepMs)}</td>
              </tr>
              <tr>
                <td>验证步时间</td>
                <td>{fmtMs(speculative.verifyStepMs)}</td>
              </tr>
              <tr>
                <td>周期时间</td>
                <td>{fmtMs(speculative.cycleTimeMs)}</td>
              </tr>
              <tr>
                <td>TPOT（投机）</td>
                <td>
                  <b>{fmtMs(speculative.verifyStepMs / speculative.expectedTokensPerCycle + speculative.draftStepMs * speculative.gamma / speculative.expectedTokensPerCycle)}</b>
                </td>
              </tr>
              <tr>
                <td>TPOT（基线）</td>
                <td>{fmtMs(speculative.baselineTpotMs)}</td>
              </tr>
              <tr>
                <td>加速比</td>
                <td>
                  <b>{speculative.speedup.toFixed(2)}×</b>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="muted small">
          投机采样通过小模型（Draft）生成候选 token，大模型（Main）并行验证，从而降低每 token
          生成时间（TPOT）。Draft 和 Main 模型同时在 GPU 内存中；Draft 仅支持 TP 并行。
        </div>
      </div>
    </div>
  );
}
