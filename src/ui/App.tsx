// Main app: left input panel + right results panel.
// All computation runs client-side against the pure core (src/core).

import { useEffect, useMemo, useState } from 'react';
import { ALL_MODELS } from '../data/models';
import { ALL_GPUS } from '../data/gpus/nvidia';
import { INTRA_NODES_CONNECTION, INTER_NODES_CONNECTION } from '../data/network';
import { evaluate } from '../core/metrics';
import { solveParallelLayout } from '../core/solver';
import type { SolverResult } from '../core/solver';
import { validateLayout } from '../core/layout';
import { PCIE_BW_GBPS } from '../core/hardware';
import {
  IDEAL,
  DEFAULT_ALPHA_INTRA_MS,
  DEFAULT_ALPHA_INTER_MS,
} from '../core/calibration';
import type { Calibration } from '../core/calibration';
import type {
  ParallelLayout,
  QuantPrecision,
  SpeculativeConfig,
  SystemSpec,
  Workload,
} from '../core/types';
import type { Result } from '../core/errors';
import type { EvaluationResult } from '../core/metrics';
import { suggestDraftModel } from '../data/models/suggest';
import { SearchSelect } from './components/SearchSelect';
import type { SearchOption } from './components/SearchSelect';
import {
  ChipGroup,
  CollapseSection,
  NumberField,
  Section,
  SelectField,
  Toggle,
} from './components/Fields';
import { Results } from './components/Results';
import { fmtCtx } from './lib/format';
import {
  CALIBRATED_PRESET,
  DEFAULT_BATCH_SIZE,
  DEFAULT_GPUS_PER_NODE,
  DEFAULT_HEADROOM,
  DEFAULT_INPUT_LEN,
  DEFAULT_INTER_NODE_BW_GBPS,
  DEFAULT_KV_TRANSFER_OVERLAP,
  DEFAULT_OUTPUT_LEN,
  DEFAULT_QUANT,
  MIN_GPUS_FOR_INTER_NODE,
  MIN_GPUS_FOR_PD_DISAGG,
  MIN_GPUS_FOR_PER_NODE,
  SD_RATIO_MIN,
  SD_RATIO_MAX,
  DEFAULT_GAMMA,
  DEFAULT_ACCEPTANCE_RATE,
} from './lib/constants';

const QUANT_OPTIONS: { value: QuantPrecision; label: string; sub: string }[] = [
  { value: 'fp32', label: 'FP32', sub: '4 Bytes' },
  { value: 'bf16', label: 'BF16', sub: '2 Bytes' },
  { value: 'fp8', label: 'FP8', sub: '1 Bytes' },
  { value: 'int8', label: 'INT8', sub: '1 Bytes' },
  { value: 'int4', label: 'INT4', sub: '0.5 Bytes' },
  { value: 'fp4', label: 'FP4', sub: '0.5 Bytes' },
];

const FALLBACK_LAYOUT: ParallelLayout = { tp: 1, pp: 1, ep: 1, dp: 1 };

function intOr(v: number, fallback: number, min: number): number {
  if (!Number.isFinite(v)) return fallback;
  return Math.max(min, Math.round(v));
}

function fracOr(v: number, fallback: number): number {
  if (!Number.isFinite(v)) return fallback;
  return Math.min(1, Math.max(0, v));
}

// Largest divisor of n that is <= cap (n, cap >= 1).
function largestDivisorAtMost(n: number, cap: number): number {
  for (let d = Math.min(cap, n); d >= 1; d--) {
    if (n % d === 0) return d;
  }
  return 1;
}

interface ComputedCore {
  spec: SystemSpec;
  cal: Calibration;
  result: Result<EvaluationResult>;
  solved: SolverResult;
  effectiveLayout: ParallelLayout;
  layoutIsOverride: boolean;
  prefillSolved: SolverResult | null;
  decodeSolved: SolverResult | null;
  prefillLayoutIsOverride: boolean;
  decodeLayoutIsOverride: boolean;
  warnings: string[];
  bMaxPlaceholder: string | undefined;
}

export default function App() {
  // --- hardware ---
  const [modelId, setModelId] = useState('llama3_1_8b');
  const [gpuId, setGpuId] = useState('h100_sxm');
  const [numGpus, setNumGpus] = useState(1);
  const [gpusPerNode, setGpusPerNode] = useState(DEFAULT_GPUS_PER_NODE);
  const [intraId, setIntraId] = useState('auto');
  const [interId, setInterId] = useState('ib_ndr');
  // --- workload / quant ---
  const [quant, setQuant] = useState<QuantPrecision>(DEFAULT_QUANT);
  const [inputLen, setInputLen] = useState(DEFAULT_INPUT_LEN);
  const [outputLen, setOutputLen] = useState(DEFAULT_OUTPUT_LEN);
  const [batchSize, setBatchSize] = useState(DEFAULT_BATCH_SIZE);
  // --- switches / parallelism ---
  const [flashAttention, setFlashAttention] = useState(true);
  const [disaggOn, setDisaggOn] = useState(false);
  const [prefillGpus, setPrefillGpus] = useState(1);
  const [decodeGpus, setDecodeGpus] = useState(1);
  const [prefillDp, setPrefillDp] = useState(1);
  const [decodeDp, setDecodeDp] = useState(1);
  const [kvOverlap, setKvOverlap] = useState(DEFAULT_KV_TRANSFER_OVERLAP);
  const [dp, setDp] = useState(1);
  const [ep, setEp] = useState(1);
  const [layoutOverride, setLayoutOverride] = useState<ParallelLayout | null>(null);
  const [prefillLayoutOverride, setPrefillLayoutOverride] = useState<ParallelLayout | null>(null);
  const [decodeLayoutOverride, setDecodeLayoutOverride] = useState<ParallelLayout | null>(null);
  // --- speculative decoding ---
  const [sdOn, setSdOn] = useState(false);
  const [draftModelId, setDraftModelId] = useState('');
  const [draftTp, setDraftTp] = useState(1);
  const [gamma, setGamma] = useState(DEFAULT_GAMMA);
  const [acceptanceRate, setAcceptanceRate] = useState(DEFAULT_ACCEPTANCE_RATE);
  // --- calibration ---
  const [calOpen, setCalOpen] = useState(false);
  const [headroom, setHeadroom] = useState(DEFAULT_HEADROOM);
  const [cal, setCal] = useState<Calibration>({ ...CALIBRATED_PRESET });

  const modelOptions = useMemo<SearchOption[]>(
    () =>
      Object.values(ALL_MODELS)
        .sort((a, b) => a.paramsB - b.paramsB)
        .map((m) => ({
          id: m.id,
          name: m.name,
          tag: m.type === 'moe' ? 'MoE' : 'Dense',
          sub: `${m.paramsB}B${m.moe ? ` · 激活 ${m.moe.activeParamsB}B` : ''} · ctx ${fmtCtx(m.maxCtx)}`,
        })),
    [],
  );

  const gpuOptions = useMemo<SearchOption[]>(
    () =>
      Object.values(ALL_GPUS)
        .sort((a, b) => b.vramGb - a.vramGb)
        .map((g) => ({
          id: g.id,
          name: g.name,
          sub: `${g.vramGb} GB · ${g.bwGbps} GB/s${g.nvlinkBwGbps ? ` · NVLink ${g.nvlinkBwGbps} GB/s` : ' · PCIe'}`,
        })),
    [],
  );

  // Draft model options: models in the SD size range (5-10x smaller than main)
  const draftModelOptions = useMemo<SearchOption[]>(() => {
    const main = ALL_MODELS[modelId];
    if (!main) return [];
    const minP = main.paramsB * SD_RATIO_MIN;
    const maxP = main.paramsB * SD_RATIO_MAX;
    return Object.values(ALL_MODELS)
      .filter((m) => m.paramsB >= minP && m.paramsB <= maxP && m.id !== main.id)
      .sort((a, b) => b.paramsB - a.paramsB)
      .map((m) => ({
        id: m.id,
        name: m.name,
        tag: m.type === 'moe' ? 'MoE' : 'Dense',
        sub: `${m.paramsB}B（主模型 ${main.paramsB}B 的 ${Math.round((m.paramsB / main.paramsB) * 100)}%）`,
      }));
  }, [modelId]);

  // Max draft TP: total GPUs or decode pool GPUs (if PD disaggregation)
  const maxDraftTp = disaggOn ? intOr(decodeGpus, 1, 1) : intOr(numGpus, 1, 1);

  const core = useMemo<ComputedCore | null>(() => {
    const model = ALL_MODELS[modelId];
    const gpu = ALL_GPUS[gpuId];
    if (!model || !gpu) return null;

    const nGpus = intOr(numGpus, 1, 1);
    // The per-node split only matters beyond 3 GPUs; smaller clusters always
    // fit in one node. Never exceed the total GPU count.
    const nPerNode =
      nGpus >= MIN_GPUS_FOR_PER_NODE
        ? Math.min(intOr(gpusPerNode, DEFAULT_GPUS_PER_NODE, 1), nGpus)
        : nGpus;
    const workload: Workload = {
      batchSize: intOr(batchSize, DEFAULT_BATCH_SIZE, 1),
      inputLen: intOr(inputLen, DEFAULT_INPUT_LEN, 1),
      outputLen: intOr(outputLen, DEFAULT_OUTPUT_LEN, 1),
    };
    const head = fracOr(headroom, DEFAULT_HEADROOM);
    const calibration: Calibration = {
      mfuPrefill: fracOr(cal.mfuPrefill, 1),
      bwEffDecode: fracOr(cal.bwEffDecode, 1),
      commEffIntra: fracOr(cal.commEffIntra, 1),
      commEffInter: fracOr(cal.commEffInter, 1),
      tpCommOverlap: fracOr(cal.tpCommOverlap, 1),
      epCommOverlap: fracOr(cal.epCommOverlap, 1),
      ppCommOverlap: fracOr(cal.ppCommOverlap, 1),
      alphaIntraMs: Math.max(0, Number.isFinite(cal.alphaIntraMs) ? cal.alphaIntraMs : 0),
      alphaInterMs: Math.max(0, Number.isFinite(cal.alphaInterMs) ? cal.alphaInterMs : 0),
    };

    const intraPreset = INTRA_NODES_CONNECTION.find((c) => c.id === intraId);
    const intraBw = intraId === 'auto' ? gpu.nvlinkBwGbps ?? PCIE_BW_GBPS : intraPreset?.bw ?? PCIE_BW_GBPS;
    const interBw =
      INTER_NODES_CONNECTION.find((c) => c.id === interId)?.bw ?? DEFAULT_INTER_NODE_BW_GBPS;

    const solverBase = {
      model,
      gpu,
      gpusPerNode: nPerNode,
      // Same bounds as the input layer: <= N, product <= N, and both divide N.
      dp: largestDivisorAtMost(nGpus, Math.min(intOr(dp, 1, 1), nGpus)),
      ep: largestDivisorAtMost(
        Math.max(1, Math.floor(nGpus / largestDivisorAtMost(nGpus, Math.min(intOr(dp, 1, 1), nGpus)))),
        Math.min(intOr(ep, 1, 1), nGpus),
      ),
      intraNodeBwGbps: intraBw,
      interNodeBwGbps: interBw,
      workload,
      weightQuant: quant,
      kvQuant: quant,
      flashAttention,
      headroom: head,
      cal: calibration,
    };

    const solved = solveParallelLayout({ ...solverBase, numGpus: nGpus });
    const autoLayout = solved.chosen ?? solved.bestEffort ?? FALLBACK_LAYOUT;
    const overrideValid =
      layoutOverride !== null &&
      validateLayout(layoutOverride, model, nGpus, nPerNode).length === 0;
    const effectiveLayout = overrideValid && layoutOverride ? layoutOverride : autoLayout;

    const warnings: string[] = [];
    if (workload.inputLen > model.maxCtx) {
      warnings.push(
        `输入长度 ${workload.inputLen} 超过模型最大上下文 ${model.maxCtx}（${fmtCtx(model.maxCtx)}）`,
      );
    }

    let prefillSolved: SolverResult | null = null;
    let decodeSolved: SolverResult | null = null;
    let disagg: SystemSpec['disagg'];
    if (disaggOn) {
      const pN = intOr(prefillGpus, Math.max(1, Math.floor(nGpus / 2)), 1);
      const dN = intOr(decodeGpus, Math.max(1, nGpus - pN), 1);
      const pDp = Math.min(intOr(prefillDp, 1, 1), pN);
      const dDp = Math.min(intOr(decodeDp, 1, 1), dN);
      prefillSolved = solveParallelLayout({ ...solverBase, numGpus: pN, dp: pDp });
      decodeSolved = solveParallelLayout({ ...solverBase, numGpus: dN, dp: dDp });

      // Validate override layouts
      const prefillOverrideValid =
        prefillLayoutOverride !== null &&
        validateLayout(prefillLayoutOverride, model, pN, nPerNode).length === 0;
      const decodeOverrideValid =
        decodeLayoutOverride !== null &&
        validateLayout(decodeLayoutOverride, model, dN, nPerNode).length === 0;

      disagg = {
        prefillGpus: pN,
        decodeGpus: dN,
        prefillLayout: prefillOverrideValid
          ? prefillLayoutOverride
          : prefillSolved.chosen ?? prefillSolved.bestEffort ?? FALLBACK_LAYOUT,
        decodeLayout: decodeOverrideValid
          ? decodeLayoutOverride
          : decodeSolved.chosen ?? decodeSolved.bestEffort ?? FALLBACK_LAYOUT,
        kvTransferOverlap: fracOr(kvOverlap, DEFAULT_KV_TRANSFER_OVERLAP),
      };
      if (nGpus < MIN_GPUS_FOR_PD_DISAGG) warnings.push('PD 分离至少需要 2 张 GPU');
      if (pN + dN > nGpus) {
        warnings.push(`Prefill（${pN}）+ Decode（${dN}）GPU 数之和 ${pN + dN} 超过总数 ${nGpus}`);
      }
    }

    const spec: SystemSpec = {
      model,
      gpu,
      gpusPerNode: nPerNode,
      interNodeBwGbps: interBw,
      ...(intraId !== 'auto' ? { intraNodeBwGbps: intraBw } : {}),
      workload,
      weightQuant: quant,
      kvQuant: quant,
      layout: effectiveLayout,
      flashAttention,
      headroom: head,
      ...(disagg ? { disagg } : {}),
    };

    // Speculative decoding: add draft model config if enabled
    if (sdOn && draftModelId && ALL_MODELS[draftModelId]) {
      const draftModel = ALL_MODELS[draftModelId];
      const maxDraftTp = disaggOn ? intOr(decodeGpus, 1, 1) : nGpus;
      const sdConfig: SpeculativeConfig = {
        draftModel,
        draftTp: Math.min(intOr(draftTp, 1, 1), maxDraftTp),
        gamma: intOr(gamma, DEFAULT_GAMMA, 1),
        acceptanceRate: fracOr(acceptanceRate, DEFAULT_ACCEPTANCE_RATE),
      };
      spec.speculative = sdConfig;
    }

    const result = evaluate(spec, calibration);

    // Calculate override flags for PD disaggregation
    const prefillLayoutIsOverride =
      disaggOn &&
      prefillLayoutOverride !== null &&
      validateLayout(prefillLayoutOverride, model, intOr(prefillGpus, 1, 1), nPerNode).length === 0;
    const decodeLayoutIsOverride =
      disaggOn &&
      decodeLayoutOverride !== null &&
      validateLayout(decodeLayoutOverride, model, intOr(decodeGpus, 1, 1), nPerNode).length === 0;

    return {
      spec,
      cal: calibration,
      result,
      solved,
      effectiveLayout,
      layoutIsOverride: overrideValid,
      prefillSolved,
      decodeSolved,
      prefillLayoutIsOverride,
      decodeLayoutIsOverride,
      warnings,
      bMaxPlaceholder: result.ok ? String(result.value.memory.bMax) : undefined,
    };
  }, [
    modelId,
    gpuId,
    numGpus,
    gpusPerNode,
    intraId,
    interId,
    quant,
    inputLen,
    outputLen,
    batchSize,
    flashAttention,
    disaggOn,
    prefillGpus,
    decodeGpus,
    prefillDp,
    decodeDp,
    kvOverlap,
    dp,
    ep,
    layoutOverride,
    prefillLayoutOverride,
    decodeLayoutOverride,
    headroom,
    cal,
    sdOn,
    draftModelId,
    draftTp,
    gamma,
    acceptanceRate,
  ]);

  const model = ALL_MODELS[modelId];
  const isMoe = model?.type === 'moe';

  // PD pool bounds: Prefill + Decode must stay within the total GPU count,
  // with at least one GPU kept free for the other pool.
  const nGpusUi = intOr(numGpus, 1, 1);
  const prefillMax = Math.max(1, nGpusUi - 1);
  const decodeMax = prefillMax;

  // DP / EP bounds: neither may exceed the GPU count, nor may their product,
  // and both must divide the cluster (DP | N, EP | N/DP). The maxima below
  // are themselves valid values.
  const dpMax = largestDivisorAtMost(nGpusUi, Math.floor(nGpusUi / intOr(ep, 1, 1)));
  const epReplica = Math.max(1, Math.floor(nGpusUi / intOr(dp, 1, 1)));
  const epMax = epReplica;

  // EP only applies to MoE models: keep it pinned to 1 for dense models.
  useEffect(() => {
    if (ALL_MODELS[modelId]?.type !== 'moe') setEp(1);
  }, [modelId]);

  // GPUs-per-node default, re-applied whenever the total changes:
  // 8 once the cluster reaches 8 GPUs, otherwise the cluster itself fits in
  // one node so the default equals the total.
  useEffect(() => {
    const n = intOr(numGpus, 1, 1);
    setGpusPerNode(Math.min(n, DEFAULT_GPUS_PER_NODE));
  }, [numGpus]);

  // DP / EP must each fit in the cluster, their product must fit, and both
  // must divide it (DP | N, EP | N/DP). Re-clamp when the total changes.
  useEffect(() => {
    const n = intOr(numGpus, 1, 1);
    const d = largestDivisorAtMost(n, Math.min(intOr(dp, 1, 1), n));
    const replica = Math.max(1, Math.floor(n / d));
    const e = largestDivisorAtMost(replica, Math.min(intOr(ep, 1, 1), replica));
    setDp(d);
    setEp(e);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [numGpus]);

  // PD disaggregation needs at least 2 GPUs; hide and disable it below that.
  // When the total changes, re-clamp the pool split into the new budget.
  useEffect(() => {
    const n = intOr(numGpus, 1, 1);
    if (n < MIN_GPUS_FOR_PD_DISAGG) {
      setDisaggOn(false);
      return;
    }
    const p = Math.min(intOr(prefillGpus, 1, 1), n - 1);
    const d = Math.min(intOr(decodeGpus, 1, 1), n - p);
    setPrefillGpus(p);
    setDecodeGpus(d);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [numGpus]);

  // Re-clamp prefill/decode DP when pool GPU counts change.
  // DP must be <= pool GPU count and must divide it.
  useEffect(() => {
    const pGpus = intOr(prefillGpus, 1, 1);
    const dGpus = intOr(decodeGpus, 1, 1);
    const pDp = largestDivisorAtMost(pGpus, Math.min(intOr(prefillDp, 1, 1), pGpus));
    const dDp = largestDivisorAtMost(dGpus, Math.min(intOr(decodeDp, 1, 1), dGpus));
    setPrefillDp(pDp);
    setDecodeDp(dDp);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prefillGpus, decodeGpus]);

  // Draft TP max depends on PD disaggregation status:
  // - PD ON: max = decode pool GPUs
  // - PD OFF: max = total GPUs
  // Re-clamp draftTp when these dependencies change.
  useEffect(() => {
    const maxDraftTp = disaggOn ? intOr(decodeGpus, 1, 1) : intOr(numGpus, 1, 1);
    setDraftTp(Math.min(intOr(draftTp, 1, 1), maxDraftTp));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [disaggOn, decodeGpus, numGpus]);

  function setCalField(k: keyof Calibration, v: number) {
    setCal((c) => ({ ...c, [k]: v }));
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>LLM 推理计算器</h1>
        <span className="app-sub">静态性能模型 · 理想值上界</span>
      </header>

      <div className="columns">
        <aside className="col-left">
          <Section title="模型与 GPU">
            <div className="field">
              <span className="field-label">模型</span>
              <SearchSelect options={modelOptions} value={modelId} onChange={setModelId} />
            </div>
            <div className="field">
              <span className="field-label">GPU</span>
              <SearchSelect options={gpuOptions} value={gpuId} onChange={setGpuId} />
            </div>
            <div className="field-grid">
              <NumberField label="GPU 数目" value={numGpus} onChange={setNumGpus} min={1} />
              {intOr(numGpus, 1, 1) >= MIN_GPUS_FOR_PER_NODE && (
                <NumberField
                  label="每节点 GPU 数"
                  value={gpusPerNode}
                  onChange={setGpusPerNode}
                  min={1}
                  max={intOr(numGpus, 1, 1)}
                  hint="总GPU数超过 8 时默认 8"
                />
              )}
            </div>
          </Section>

          <Section title="互连">
            <SelectField
              label="节点内互连（intra-node）"
              value={intraId}
              onChange={setIntraId}
              options={[
                { value: 'auto', label: '自动（按 GPU 规格：NVLink 或 PCIe 回退）' },
                ...INTRA_NODES_CONNECTION.map((c) => ({
                  value: c.id,
                  label: `${c.label} — ${c.bw} GB/s`,
                })),
              ]}
            />
            {intOr(numGpus, 1, 1) >= MIN_GPUS_FOR_INTER_NODE && (
              <SelectField
                label="节点间互连（inter-node）"
                value={interId}
                onChange={setInterId}
                options={INTER_NODES_CONNECTION.map((c) => ({
                  value: c.id,
                  label: `${c.label} — ${c.bw} GB/s`,
                }))}
                hint="仅多节点（跨节点 PP / PD KV 传输）时生效"
              />
            )}
          </Section>

          <Section title="量化与负载">
            <ChipGroup
              label="量化（权重与 KV cache 统一）"
              value={quant}
              onChange={setQuant}
              options={QUANT_OPTIONS}
            />
            <div className="field-grid">
              <NumberField
                label="输入长度"
                value={inputLen}
                onChange={setInputLen}
                min={1}
                error={
                  model && intOr(inputLen, DEFAULT_INPUT_LEN, 1) > model.maxCtx
                    ? `超过模型最大上下文 ${model.maxCtx}`
                    : undefined
                }
              />
              <NumberField label="输出长度" value={outputLen} onChange={setOutputLen} min={1} />
            </div>
            <NumberField
              label="Batch size"
              value={batchSize}
              onChange={setBatchSize}
              min={1}
              placeholder={core?.bMaxPlaceholder}
              hint="占位符为当前配置下显存允许的最大 batch size"
            />
          </Section>

          <Section title="并行与开关">
            <div className="field-grid">
              <NumberField
                label="DP"
                value={dp}
                onChange={(v) =>
                  setDp(largestDivisorAtMost(nGpusUi, Math.min(intOr(v, 1, 1), dpMax)))
                }
                min={1}
                max={dpMax}
              />
              <NumberField
                label="EP"
                value={ep}
                onChange={(v) =>
                  setEp(largestDivisorAtMost(epReplica, Math.min(intOr(v, 1, 1), epMax)))
                }
                min={1}
                max={epMax}
                disabled={!isMoe}
                hint={!isMoe ? 'Dense 模型不支持 EP' : undefined}
              />
            </div>
            <div className="muted small" style={{ marginBottom: 8 }}>
              TP / PP 由求解器自动决定（DP ⇒ TP ⇒ EP ⇒ PP，优先保证显存装下）；
              DP、EP 及两者乘积均不超过 GPU 总数，且须整除总数（非法值自动吸附到最近的合法约数）
            </div>
            <div className="toggle-stack">
              <Toggle
                label="FlashAttention"
                desc="开启后激活显存按 O(N·h) 计，避免 N² 注意力矩阵"
                checked={flashAttention}
                onChange={setFlashAttention}
              />
              {intOr(numGpus, 1, 1) >= MIN_GPUS_FOR_PD_DISAGG && (
                <Toggle
                  label="PD 分离（Prefill-Decode Disaggregation）"
                  desc="Prefill 与 Decode 使用独立 GPU 池"
                  checked={disaggOn}
                  onChange={(v) => {
                    setDisaggOn(v);
                    if (v) {
                      const n = intOr(numGpus, 1, 1);
                      const p = Math.max(1, Math.floor(n / 2));
                      setPrefillGpus(p);
                      setDecodeGpus(Math.max(1, n - p));
                    }
                  }}
                />
              )}
            </div>
            {disaggOn && intOr(numGpus, 1, 1) >= MIN_GPUS_FOR_PD_DISAGG && (
              <div className="pd-extra">
                <div className="field-grid">
                  <NumberField
                    label="Prefill GPU 数"
                    value={prefillGpus}
                    onChange={(v) => {
                      setPrefillGpus(Math.min(intOr(v, 1, 1), prefillMax))
                      setDecodeGpus(numGpus - v)
                    }}
                    min={1}
                    max={prefillMax}
                  />
                  <NumberField
                    label="Decode GPU 数"
                    value={decodeGpus}
                    onChange={(v) => {
                      setDecodeGpus(Math.min(intOr(v, 1, 1), decodeMax))
                      setPrefillGpus(numGpus - v)
                    }}
                    min={1}
                    max={decodeMax}
                  />
                </div>
                <div className="field-grid">
                  <NumberField
                    label="Prefill DP"
                    value={prefillDp}
                    onChange={(v) => setPrefillDp(Math.min(intOr(v, 1, 1), intOr(prefillGpus, 1, 1)))}
                    min={1}
                    max={intOr(prefillGpus, 1, 1)}
                    hint="Prefill 池的数据并行度"
                  />
                  <NumberField
                    label="Decode DP"
                    value={decodeDp}
                    onChange={(v) => setDecodeDp(Math.min(intOr(v, 1, 1), intOr(decodeGpus, 1, 1)))}
                    min={1}
                    max={intOr(decodeGpus, 1, 1)}
                    hint="Decode 池的数据并行度（高 DP 可最大化 TP）"
                  />
                </div>
                <div className="muted small">
                  两池之和须 ≤ GPU 总数（{nGpusUi}），允许部分 GPU 不分配。DP 须整除对应池的 GPU 数。
                </div>
                <div className="field">
                  <span className="field-label">
                    KV 传输重叠系数 — {(fracOr(kvOverlap, DEFAULT_KV_TRANSFER_OVERLAP)).toFixed(2)}
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={fracOr(kvOverlap, DEFAULT_KV_TRANSFER_OVERLAP)}
                    onChange={(e) => setKvOverlap(e.target.valueAsNumber)}
                  />
                  <span className="field-hint">
                    KV 传输时间与 prefill 计算重叠（被隐藏）的比例：0 = 完全串行暴露在
                    TTFT 中，1 = 完全隐藏；有逐层流水的现代引擎约 0.8 ~ 1
                  </span>
                </div>
              </div>
            )}
          </Section>

          <Section title="投机采样 (Speculative Decoding)">
            <Toggle
              label="启用投机采样"
              desc="小模型草稿 + 大模型验证，降低 TPOT"
              checked={sdOn}
              onChange={(v) => {
                setSdOn(v);
                if (v && !draftModelId) {
                  // Auto-suggest draft model on first enable
                  const main = ALL_MODELS[modelId];
                  if (main) {
                    const suggestion = suggestDraftModel(main, ALL_MODELS, SD_RATIO_MIN, SD_RATIO_MAX);
                    if (suggestion) {
                      setDraftModelId(suggestion.id);
                      // Default draft TP = main TP, capped by max
                      const mainTp = core?.effectiveLayout.tp ?? 1;
                      setDraftTp(Math.min(mainTp, maxDraftTp));
                    }
                  }
                }
              }}
            />
            {sdOn && (
              <div className="sd-extra">
                <div className="field">
                  <span className="field-label">草稿模型（Draft Model）</span>
                  <SearchSelect
                    options={draftModelOptions}
                    value={draftModelId}
                    onChange={setDraftModelId}
                    placeholder={draftModelOptions.length > 0 ? '选择草稿模型' : '无可用草稿模型'}
                  />
                  {draftModelOptions.length === 0 && (
                    <span className="field-hint err-hint">
                      模型库中没有适合当前主模型的草稿模型（需要 {Math.round(ALL_MODELS[modelId]?.paramsB ?? 0 * SD_RATIO_MIN)}B
                      ~ {Math.round(ALL_MODELS[modelId]?.paramsB ?? 0 * SD_RATIO_MAX)}B 的 dense 模型）
                    </span>
                  )}
                </div>
                <NumberField
                  label="Draft TP"
                  value={draftTp}
                  onChange={(v) => setDraftTp(Math.min(intOr(v, 1, 1), maxDraftTp))}
                  min={1}
                  max={maxDraftTp}
                  hint="草稿模型的张量并行度（仅支持 TP，默认跟随主模型 TP）"
                />
                <div className="field-grid">
                  <NumberField
                    label="γ（草稿步数）"
                    value={gamma}
                    onChange={setGamma}
                    min={1}
                    max={16}
                    hint="每次验证前草稿模型生成的 token 数，典型 4–8"
                  />
                  <NumberField
                    label="接受率"
                    value={acceptanceRate}
                    onChange={setAcceptanceRate}
                    min={0}
                    max={1}
                    step={0.05}
                    hint="草稿 token 被主模型接受的概率；取决于模型配对，通常 0.5–0.9"
                  />
                </div>
                <div className="muted small">
                  Draft 和 Main 模型同时在 GPU 内存中；Draft 仅支持 TP 并行，不支持 PP/EP/DP。
                  Draft TP 默认跟随主模型 TP，最大值 = {disaggOn ? 'Decode' : ''} GPU 总数（{maxDraftTp}）。
                </div>
              </div>
            )}
          </Section>

          <CollapseSection
            title="校准参数（高级）"
            summary={calOpen ? undefined : '默认：校准值（calibration/README.md 锚点）'}
            open={calOpen}
            onToggle={() => setCalOpen((v) => !v)}
          >
            <div className="field-grid">
              <NumberField
                label="显存预留 headroom"
                value={headroom}
                onChange={setHeadroom}
                min={0}
                step={0.05}
              />
              <NumberField
                label="MFU_prefill"
                value={cal.mfuPrefill}
                onChange={(v) => setCalField('mfuPrefill', v)}
                min={0}
                step={0.05}
              />
              <NumberField
                label="BW_eff_decode"
                value={cal.bwEffDecode}
                onChange={(v) => setCalField('bwEffDecode', v)}
                min={0}
                step={0.05}
              />
              <NumberField
                label="通信效率（节点内）"
                value={cal.commEffIntra}
                onChange={(v) => setCalField('commEffIntra', v)}
                min={0}
                step={0.05}
              />
              <NumberField
                label="通信效率（节点间）"
                value={cal.commEffInter}
                onChange={(v) => setCalField('commEffInter', v)}
                min={0}
                step={0.05}
              />
              <NumberField
                label="TP 通信重叠"
                value={cal.tpCommOverlap}
                onChange={(v) => setCalField('tpCommOverlap', v)}
                min={0}
                step={0.05}
              />
              <NumberField
                label="EP 通信重叠"
                value={cal.epCommOverlap}
                onChange={(v) => setCalField('epCommOverlap', v)}
                min={0}
                step={0.05}
              />
              <NumberField
                label="PP 通信重叠"
                value={cal.ppCommOverlap}
                onChange={(v) => setCalField('ppCommOverlap', v)}
                min={0}
                step={0.05}
              />
              <NumberField
                label="α 节点内 (ms)"
                value={cal.alphaIntraMs}
                onChange={(v) => setCalField('alphaIntraMs', v)}
                min={0}
                step={0.01}
                placeholder={String(DEFAULT_ALPHA_INTRA_MS)}
              />
              <NumberField
                label="α 节点间 (ms)"
                value={cal.alphaInterMs}
                onChange={(v) => setCalField('alphaInterMs', v)}
                min={0}
                step={0.01}
                placeholder={String(DEFAULT_ALPHA_INTER_MS)}
              />
            </div>
            <div className="btn-row">
              <button
                type="button"
                className="btn"
                title="性能上界：全部效率 = 1，仅显存预留保持 0.1"
                onClick={() => {
                  setCal({ ...IDEAL });
                  setHeadroom(DEFAULT_HEADROOM);
                }}
              >
                重置为理想值
              </button>
              <button
                type="button"
                className="btn"
                title="按 calibration/README.md 锚点取最大值"
                onClick={() => {
                  setCal({ ...CALIBRATED_PRESET });
                  setHeadroom(DEFAULT_HEADROOM);
                }}
              >
                重置为校准值
              </button>
            </div>
          </CollapseSection>
        </aside>

        <main className="col-right">
          {core && model ? (
            <Results
              model={model}
              gpu={ALL_GPUS[gpuId]!}
              spec={core.spec}
              cal={core.cal}
              result={core.result}
              solved={core.solved}
              effectiveLayout={core.effectiveLayout}
              layoutIsOverride={core.layoutIsOverride}
              onPickLayout={setLayoutOverride}
              disaggOn={disaggOn}
              prefillSolved={core.prefillSolved}
              decodeSolved={core.decodeSolved}
              prefillLayoutIsOverride={core.prefillLayoutIsOverride}
              decodeLayoutIsOverride={core.decodeLayoutIsOverride}
              onPickPrefillLayout={setPrefillLayoutOverride}
              onPickDecodeLayout={setDecodeLayoutOverride}
              warnings={core.warnings}
            />
          ) : (
            <div className="card">无效配置</div>
          )}
        </main>
      </div>
    </div>
  );
}
