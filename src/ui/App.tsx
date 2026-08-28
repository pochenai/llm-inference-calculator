// Main app: left input panel + right results panel.
// All computation runs client-side against the pure core (src/core).

import { useEffect, useMemo, useState } from 'react';
import { ALL_MODELS, MODEL_SIZE_TIER, SIZE_TIERS } from '../data/models';
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
import { LocaleProvider, useI18n } from './lib/i18n';
import {
  CALIBRATED_PRESET,
  DEFAULT_BATCH_SIZE,
  DEFAULT_GPUS_PER_NODE,
  DEFAULT_HEADROOM,
  DEFAULT_INPUT_LEN,
  DEFAULT_INTER_NODE_BW_GBPS,
  DEFAULT_KV_TRANSFER_OVERLAP,
  DEFAULT_OUTPUT_LEN,
  DEFAULT_PREFILL_RATIO,
  DEFAULT_QUANT,
  MIN_GPUS_FOR_INTER_NODE,
  MIN_GPUS_FOR_PD_DISAGG,
  MIN_GPUS_FOR_PER_NODE,
  SD_RATIO_MIN,
  SD_RATIO_MAX,
  DEFAULT_GAMMA,
  DEFAULT_ACCEPTANCE_RATE,
} from './lib/constants';
import { useUrlState, writeLocaleToUrl } from './lib/useUrlParams';

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
  return (
    <LocaleProvider>
      <AppContent />
    </LocaleProvider>
  );
}

function AppContent() {
  const { locale, setLocale, t } = useI18n();

  // --- all persistent state from URL params ---
  const [state, setState] = useUrlState();
  const {
    modelId, gpuId, numGpus, gpusPerNode, intraId, interId,
    quant, inputLen, outputLen, batchSize, prefillRatio, prefillRatioOn,
    flashAttention, disaggOn, prefillGpus, decodeGpus, prefillDp, decodeDp,
    kvOverlap, dp, ep, layoutOverride, prefillLayoutOverride, decodeLayoutOverride,
    sdOn, draftModelId, draftTp, gamma, acceptanceRate,
    headroom, cal,
  } = state;

  // UI-only state (not persisted in URL)
  const [calOpen, setCalOpen] = useState(false);

  // Convenience setters that patch individual fields.
  const setModelId = (v: string) => setState((s) => ({ ...s, modelId: v }));
  const setGpuId = (v: string) => setState((s) => ({ ...s, gpuId: v }));
  const setNumGpus = (v: number) => setState((s) => ({ ...s, numGpus: v }));
  const setGpusPerNode = (v: number) => setState((s) => ({ ...s, gpusPerNode: v }));
  const setIntraId = (v: string) => setState((s) => ({ ...s, intraId: v }));
  const setInterId = (v: string) => setState((s) => ({ ...s, interId: v }));
  const setQuant = (v: QuantPrecision) => setState((s) => ({ ...s, quant: v }));
  const setInputLen = (v: number) => setState((s) => ({ ...s, inputLen: v }));
  const setOutputLen = (v: number) => setState((s) => ({ ...s, outputLen: v }));
  const setBatchSize = (v: number) => setState((s) => ({ ...s, batchSize: v }));
  const setPrefillRatio = (v: number | ((prev: number) => number)) =>
    setState((s) => ({ ...s, prefillRatio: typeof v === 'function' ? v(s.prefillRatio) : v }));
  const setPrefillRatioOn = (v: boolean) => setState((s) => ({ ...s, prefillRatioOn: v }));
  const setFlashAttention = (v: boolean) => setState((s) => ({ ...s, flashAttention: v }));
  const setDisaggOn = (v: boolean) => setState((s) => ({ ...s, disaggOn: v }));
  const setPrefillGpus = (v: number) => setState((s) => ({ ...s, prefillGpus: v }));
  const setDecodeGpus = (v: number) => setState((s) => ({ ...s, decodeGpus: v }));
  const setPrefillDp = (v: number) => setState((s) => ({ ...s, prefillDp: v }));
  const setDecodeDp = (v: number) => setState((s) => ({ ...s, decodeDp: v }));
  const setKvOverlap = (v: number) => setState((s) => ({ ...s, kvOverlap: v }));
  const setDp = (v: number) => setState((s) => ({ ...s, dp: v }));
  const setEp = (v: number) => setState((s) => ({ ...s, ep: v }));
  const setLayoutOverride = (v: ParallelLayout | null) => setState((s) => ({ ...s, layoutOverride: v }));
  const setPrefillLayoutOverride = (v: ParallelLayout | null) => setState((s) => ({ ...s, prefillLayoutOverride: v }));
  const setDecodeLayoutOverride = (v: ParallelLayout | null) => setState((s) => ({ ...s, decodeLayoutOverride: v }));
  const setSdOn = (v: boolean) => setState((s) => ({ ...s, sdOn: v }));
  const setDraftModelId = (v: string) => setState((s) => ({ ...s, draftModelId: v }));
  const setDraftTp = (v: number) => setState((s) => ({ ...s, draftTp: v }));
  const setGamma = (v: number) => setState((s) => ({ ...s, gamma: v }));
  const setAcceptanceRate = (v: number) => setState((s) => ({ ...s, acceptanceRate: v }));
  const setHeadroom = (v: number) => setState((s) => ({ ...s, headroom: v }));
  const setCal = (v: Calibration | ((prev: Calibration) => Calibration)) =>
    setState((s) => ({ ...s, cal: typeof v === 'function' ? v(s.cal) : v }));

  // Stable calibration reference — only changes when calibration values actually
  // change, preventing the heavy core memo from recalculating on unrelated state updates.
  const calStable = useMemo(() => cal, [JSON.stringify(cal)]);

  const modelOptions = useMemo<SearchOption[]>(
    () =>
      Object.values(ALL_MODELS)
        .sort((a, b) => a.paramsB - b.paramsB)
        .map((m) => {
          const tier = MODEL_SIZE_TIER.get(m.id);
          const activePart = m.moe
            ? ` · ${t('model.sub.moe', { activeB: m.moe.activeParamsB })}`
            : '';
          return {
            id: m.id,
            name: m.name,
            tag: m.type === 'moe' ? 'MoE' : 'Dense',
            sub: `${m.paramsB}B${activePart} · ctx ${fmtCtx(m.maxCtx)}`,
            ...(tier ? { category: tier } : {}),
          };
        }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [locale],
  );

  // Only show size tiers that have at least one model in the catalog.
  const modelSizeCategories = useMemo(() => {
    const used = new Set(modelOptions.map((o) => o.category));
    return SIZE_TIERS.filter((t) => used.has(t.value)).map((t) => ({
      value: t.value,
      label: t.label,
    }));
  }, [modelOptions]);

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
        sub: t('model.sub.draft', {
          draftB: m.paramsB,
          mainB: main.paramsB,
          pct: Math.round((m.paramsB / main.paramsB) * 100),
        }),
      }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelId, locale]);

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
      ...(prefillRatioOn
        ? { prefillRatio: Math.max(0.1, Math.min(0.9, Number.isFinite(prefillRatio) ? prefillRatio : DEFAULT_PREFILL_RATIO)) }
        : {}),
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
        t('warn.input_exceeds_ctx', { inputLen: workload.inputLen, maxCtx: model.maxCtx, ctxStr: fmtCtx(model.maxCtx) }),
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
      // Build pool-specific workloads with split batch sizes for the solver.
      // Without this, the solver evaluates layouts against the FULL batch size,
      // causing false "infeasible" verdicts for the (smaller) prefill pool.
      // When steady-state is disabled (prefillRatio undefined), both pools
      // use the full batch.
      const pBatch = workload.prefillRatio !== undefined
        ? Math.max(1, Math.round(workload.prefillRatio * workload.batchSize))
        : workload.batchSize;
      const dBatch = workload.prefillRatio !== undefined
        ? Math.max(1, Math.round((1 - workload.prefillRatio) * workload.batchSize))
        : workload.batchSize;
      const prefillWorkload: Workload = { ...workload, batchSize: pBatch };
      const decodeWorkload: Workload = { ...workload, batchSize: dBatch };
      prefillSolved = solveParallelLayout({ ...solverBase, numGpus: pN, dp: pDp, workload: prefillWorkload, pdMode: 'prefill' as const });
      decodeSolved = solveParallelLayout({ ...solverBase, numGpus: dN, dp: dDp, workload: decodeWorkload, pdMode: 'decode' as const });

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
      if (nGpus < MIN_GPUS_FOR_PD_DISAGG) warnings.push(t('warn.pd_min_gpus'));
      if (pN + dN > nGpus) {
        warnings.push(t('warn.pd_exceeds_total', { pN, dN, total: nGpus }));
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
    prefillRatio,
    prefillRatioOn,
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
    calStable,
    sdOn,
    draftModelId,
    draftTp,
    gamma,
    acceptanceRate,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    locale,
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

  // Sync locale to URL when changed.
  useEffect(() => {
    writeLocaleToUrl(locale);
  }, [locale]);

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

  const decodePct = Number.isFinite(prefillRatio)
    ? (100 - prefillRatio * 100).toFixed(0)
    : '—';

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-left">
          <h1>{t('header.title')}</h1>
          <span className="app-sub">{t('header.subtitle')}</span>
        </div>
        <div className="app-header-right">
          <select
            className="locale-select"
            value={locale}
            onChange={(e) => setLocale(e.target.value as 'en' | 'zh')}
            aria-label="Language"
          >
            <option value="en">EN</option>
            <option value="zh">中文</option>
          </select>
          <a
            className="gh-link"
            href="https://github.com/pochenai/llm-inference-calculator"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="GitHub repository"
          >
            {/* Inline GitHub Octocat mark */}
            <svg
              className="gh-icon"
              viewBox="0 0 16 16"
              width="18"
              height="18"
              aria-hidden="true"
              focusable="false"
            >
              <path
                fill="currentColor"
                d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8Z"
              />
            </svg>
            <span className="gh-text">Please⭐</span>
          </a>
        </div>
      </header>

      <div className="columns">
        <aside className="col-left">
          <Section title={t('section.model_gpu')}>
            <div className="field">
              <span className="field-label">{t('label.model')}</span>
              <SearchSelect
                options={modelOptions}
                value={modelId}
                onChange={setModelId}
                categories={modelSizeCategories}
                copyText={ALL_MODELS[modelId]?.name ?? modelId}
              />
            </div>
            <div className="field">
              <span className="field-label">{t('label.gpu')}</span>
              <SearchSelect
                options={gpuOptions}
                value={gpuId}
                onChange={setGpuId}
                copyText={ALL_GPUS[gpuId]?.name ?? gpuId}
              />
            </div>
            <div className="field-grid">
              <NumberField label={t('label.gpu_count')} value={numGpus} onChange={setNumGpus} min={1} />
              {intOr(numGpus, 1, 1) >= MIN_GPUS_FOR_PER_NODE && (
                <NumberField
                  label={t('label.gpus_per_node')}
                  value={gpusPerNode}
                  onChange={setGpusPerNode}
                  min={1}
                  max={intOr(numGpus, 1, 1)}
                  hint={t('hint.gpus_per_node')}
                />
              )}
            </div>
          </Section>

          <Section title={t('section.interconnect')}>
            <SelectField
              label={t('label.intra_node')}
              value={intraId}
              onChange={setIntraId}
              options={[
                { value: 'auto', label: t('option.intra_auto') },
                ...INTRA_NODES_CONNECTION.map((c) => ({
                  value: c.id,
                  label: `${c.label} — ${c.bw} GB/s`,
                })),
              ]}
            />
            {intOr(numGpus, 1, 1) >= MIN_GPUS_FOR_INTER_NODE && (
              <SelectField
                label={t('label.inter_node')}
                value={interId}
                onChange={setInterId}
                options={INTER_NODES_CONNECTION.map((c) => ({
                  value: c.id,
                  label: `${c.label} — ${c.bw} GB/s`,
                }))}
                hint={t('hint.inter_node')}
              />
            )}
          </Section>

          <Section title={t('section.quant_workload')}>
            <ChipGroup
              label={t('label.quantization')}
              value={quant}
              onChange={setQuant}
              options={QUANT_OPTIONS}
            />
            <div className="field-grid">
              <NumberField
                label={t('label.input_length')}
                value={inputLen}
                onChange={setInputLen}
                min={1}
                error={
                  model && intOr(inputLen, DEFAULT_INPUT_LEN, 1) > model.maxCtx
                    ? `${t('error.exceeds_context')} ${model.maxCtx}`
                    : undefined
                }
              />
              <NumberField label={t('label.output_length')} value={outputLen} onChange={setOutputLen} min={1} />
            </div>
            <NumberField
              label={t('label.batch_size')}
              value={batchSize}
              onChange={setBatchSize}
              min={1}
              placeholder={core?.bMaxPlaceholder}
              hint=""
            />
            <Toggle
              label={t('label.prefill_ratio_toggle')}
              desc={t('desc.prefill_ratio_toggle')}
              checked={prefillRatioOn}
              onChange={setPrefillRatioOn}
            />
            {prefillRatioOn && (
              <NumberField
                label={t('label.prefill_ratio')}
                value={prefillRatio}
                onChange={setPrefillRatio}
                onBlur={() =>
                  setPrefillRatio((v) =>
                    Number.isFinite(v) ? Math.max(0.1, Math.min(0.9, v)) : DEFAULT_PREFILL_RATIO,
                  )
                }
                min={0.1}
                max={0.9}
                step={0.05}
                hint={t('hint.prefill_ratio', { decodePct })}
              />
            )}
          </Section>

          <Section title={t('section.parallel_switches')}>
            <div className="field-grid">
              <NumberField
                label={t('label.dp')}
                value={dp}
                onChange={(v) =>
                  setDp(largestDivisorAtMost(nGpusUi, Math.min(intOr(v, 1, 1), dpMax)))
                }
                min={1}
                max={dpMax}
              />
              <NumberField
                label={t('label.ep')}
                value={ep}
                onChange={(v) =>
                  setEp(largestDivisorAtMost(epReplica, Math.min(intOr(v, 1, 1), epMax)))
                }
                min={1}
                max={epMax}
                disabled={!isMoe}
                hint={!isMoe ? t('hint.ep_disabled') : undefined}
              />
            </div>
            <div className="muted small" style={{ marginBottom: 8 }}>
              {t('note.parallelism')}
            </div>
            <div className="toggle-stack">
              <Toggle
                label={t('label.flash_attention')}
                desc={t('desc.flash_attention')}
                checked={flashAttention}
                onChange={setFlashAttention}
              />
              {intOr(numGpus, 1, 1) >= MIN_GPUS_FOR_PD_DISAGG && (
                <Toggle
                  label={t('label.pd_disagg')}
                  desc={t('desc.pd_disagg')}
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
                    label={t('label.prefill_gpus')}
                    value={prefillGpus}
                    onChange={(v) => {
                      setPrefillGpus(Math.min(intOr(v, 1, 1), prefillMax))
                      setDecodeGpus(numGpus - v)
                    }}
                    min={1}
                    max={prefillMax}
                  />
                  <NumberField
                    label={t('label.decode_gpus')}
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
                    label={t('label.prefill_dp')}
                    value={prefillDp}
                    onChange={(v) => setPrefillDp(Math.min(intOr(v, 1, 1), intOr(prefillGpus, 1, 1)))}
                    min={1}
                    max={intOr(prefillGpus, 1, 1)}
                    hint={t('hint.prefill_dp')}
                  />
                  <NumberField
                    label={t('label.decode_dp')}
                    value={decodeDp}
                    onChange={(v) => setDecodeDp(Math.min(intOr(v, 1, 1), intOr(decodeGpus, 1, 1)))}
                    min={1}
                    max={intOr(decodeGpus, 1, 1)}
                    hint={t('hint.decode_dp')}
                  />
                </div>
                <div className="muted small">
                  {t('note.pd_constraint', { total: nGpusUi })}
                </div>
                <div className="field">
                  <span className="field-label">
                    {t('label.kv_overlap', { value: fracOr(kvOverlap, DEFAULT_KV_TRANSFER_OVERLAP).toFixed(2) })}
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
                    {t('hint.kv_overlap')}
                  </span>
                </div>
              </div>
            )}
          </Section>

          <Section title={t('section.speculative')}>
            <Toggle
              label={t('label.sd_toggle')}
              desc={t('desc.sd_toggle')}
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
                  <span className="field-label">{t('label.draft_model')}</span>
                  <SearchSelect
                    options={draftModelOptions}
                    value={draftModelId}
                    onChange={setDraftModelId}
                    placeholder={draftModelOptions.length > 0 ? t('placeholder.select_draft') : t('placeholder.no_draft')}
                  />
                  {draftModelOptions.length === 0 && (
                    <span className="field-hint err-hint">
                      {t('error.no_draft_model', {
                        minB: Math.round((ALL_MODELS[modelId]?.paramsB ?? 0) * SD_RATIO_MIN),
                        maxB: Math.round((ALL_MODELS[modelId]?.paramsB ?? 0) * SD_RATIO_MAX),
                      })}
                    </span>
                  )}
                </div>
                <NumberField
                  label={t('label.draft_tp')}
                  value={draftTp}
                  onChange={(v) => setDraftTp(Math.min(intOr(v, 1, 1), maxDraftTp))}
                  min={1}
                  max={maxDraftTp}
                  hint={t('hint.draft_tp')}
                />
                <div className="field-grid">
                  <NumberField
                    label={t('label.gamma')}
                    value={gamma}
                    onChange={setGamma}
                    min={1}
                    max={16}
                    hint={t('hint.gamma')}
                  />
                  <NumberField
                    label={t('label.acceptance_rate')}
                    value={acceptanceRate}
                    onChange={setAcceptanceRate}
                    min={0}
                    max={1}
                    step={0.05}
                    hint={t('hint.acceptance_rate')}
                  />
                </div>
                <div className="muted small">
                  {t('note.sd_constraint', { maxDraftTp, disaggOn })}
                </div>
              </div>
            )}
          </Section>

          <CollapseSection
            title={t('section.calibration')}
            summary={calOpen ? undefined : t('summary.calibration')}
            open={calOpen}
            onToggle={() => setCalOpen((v) => !v)}
          >
            <div className="field-grid">
              <NumberField
                label={t('label.headroom')}
                value={headroom}
                onChange={setHeadroom}
                min={0}
                step={0.05}
              />
              <NumberField
                label={t('label.mfu')}
                hint={t('hint.mfu')}
                value={cal.mfuPrefill}
                onChange={(v) => setCalField('mfuPrefill', v)}
                min={0}
                step={0.05}
              />
              <NumberField
                label={t('label.bw_eff')}
                hint={t('hint.bw_eff')}
                value={cal.bwEffDecode}
                onChange={(v) => setCalField('bwEffDecode', v)}
                min={0}
                step={0.05}
              />
              <NumberField
                label={t('label.comm_eff_intra')}
                value={cal.commEffIntra}
                onChange={(v) => setCalField('commEffIntra', v)}
                min={0}
                step={0.05}
              />
              <NumberField
                label={t('label.comm_eff_inter')}
                value={cal.commEffInter}
                onChange={(v) => setCalField('commEffInter', v)}
                min={0}
                step={0.05}
              />
              <NumberField
                label={t('label.tp_comm_overlap')}
                value={cal.tpCommOverlap}
                onChange={(v) => setCalField('tpCommOverlap', v)}
                min={0}
                step={0.05}
              />
              <NumberField
                label={t('label.ep_comm_overlap')}
                value={cal.epCommOverlap}
                onChange={(v) => setCalField('epCommOverlap', v)}
                min={0}
                step={0.05}
              />
              <NumberField
                label={t('label.pp_comm_overlap')}
                value={cal.ppCommOverlap}
                onChange={(v) => setCalField('ppCommOverlap', v)}
                min={0}
                step={0.05}
              />
              <NumberField
                label={t('label.alpha_intra')}
                value={cal.alphaIntraMs}
                onChange={(v) => setCalField('alphaIntraMs', v)}
                min={0}
                step={0.01}
                placeholder={String(DEFAULT_ALPHA_INTRA_MS)}
              />
              <NumberField
                label={t('label.alpha_inter')}
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
                title={t('title.reset_ideal')}
                onClick={() => {
                  setCal({ ...IDEAL });
                  setHeadroom(DEFAULT_HEADROOM);
                }}
              >
                {t('btn.reset_ideal')}
              </button>
              <button
                type="button"
                className="btn"
                title={t('title.reset_calibrated')}
                onClick={() => {
                  setCal({ ...CALIBRATED_PRESET });
                  setHeadroom(DEFAULT_HEADROOM);
                }}
              >
                {t('btn.reset_calibrated')}
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
            <div className="card">{t('label.invalid_config')}</div>
          )}
        </main>
      </div>
    </div>
  );
}
