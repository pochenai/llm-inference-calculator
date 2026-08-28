// Syncs UI state to URL query parameters for shareability and persistence.
// Only non-default values appear in the URL, keeping it compact.

import { useEffect, useState } from 'react';
import type { ParallelLayout, QuantPrecision } from '../../core/types';
import type { Calibration } from '../../core/calibration';
import {
  CALIBRATED_PRESET,
  DEFAULT_ACCEPTANCE_RATE,
  DEFAULT_BATCH_SIZE,
  DEFAULT_GAMMA,
  DEFAULT_GPUS_PER_NODE,
  DEFAULT_HEADROOM,
  DEFAULT_INPUT_LEN,
  DEFAULT_KV_TRANSFER_OVERLAP,
  DEFAULT_OUTPUT_LEN,
  DEFAULT_PREFILL_RATIO,
  DEFAULT_QUANT,
} from './constants';

// All UI state that gets persisted in the URL.
export interface UrlState {
  // Hardware
  modelId: string;
  gpuId: string;
  numGpus: number;
  gpusPerNode: number;
  intraId: string;
  interId: string;
  // Workload / quant
  quant: QuantPrecision;
  inputLen: number;
  outputLen: number;
  batchSize: number;
  prefillRatioOn: boolean;
  prefillRatio: number;
  // Parallelism / switches
  flashAttention: boolean;
  disaggOn: boolean;
  prefillGpus: number;
  decodeGpus: number;
  prefillDp: number;
  decodeDp: number;
  kvOverlap: number;
  dp: number;
  ep: number;
  layoutOverride: ParallelLayout | null;
  prefillLayoutOverride: ParallelLayout | null;
  decodeLayoutOverride: ParallelLayout | null;
  // Speculative decoding
  sdOn: boolean;
  draftModelId: string;
  draftTp: number;
  gamma: number;
  acceptanceRate: number;
  // Calibration
  headroom: number;
  cal: Calibration;
}

export const URL_DEFAULTS: UrlState = {
  modelId: 'llama3_1_8b',
  gpuId: 'h100_sxm',
  numGpus: 1,
  gpusPerNode: DEFAULT_GPUS_PER_NODE,
  intraId: 'auto',
  interId: 'ib_ndr',
  quant: DEFAULT_QUANT,
  inputLen: DEFAULT_INPUT_LEN,
  outputLen: DEFAULT_OUTPUT_LEN,
  batchSize: DEFAULT_BATCH_SIZE,
  prefillRatioOn: false,
  prefillRatio: DEFAULT_PREFILL_RATIO,
  flashAttention: true,
  disaggOn: false,
  prefillGpus: 1,
  decodeGpus: 1,
  prefillDp: 1,
  decodeDp: 1,
  kvOverlap: DEFAULT_KV_TRANSFER_OVERLAP,
  dp: 1,
  ep: 1,
  layoutOverride: null,
  prefillLayoutOverride: null,
  decodeLayoutOverride: null,
  sdOn: false,
  draftModelId: '',
  draftTp: 1,
  gamma: DEFAULT_GAMMA,
  acceptanceRate: DEFAULT_ACCEPTANCE_RATE,
  headroom: DEFAULT_HEADROOM,
  cal: { ...CALIBRATED_PRESET },
};

// ---------------------------------------------------------------------------
// Layout serialisation: "tp-pp-ep-dp" compact string
// ---------------------------------------------------------------------------

function serializeLayout(layout: ParallelLayout): string {
  return `${layout.tp}-${layout.pp}-${layout.ep}-${layout.dp}`;
}

function parseLayout(str: string): ParallelLayout | null {
  const parts = str.split('-').map(Number);
  if (parts.length !== 4 || parts.some((v) => !Number.isFinite(v) || v < 1)) return null;
  const [tp, pp, ep, dp] = parts;
  return { tp: tp!, pp: pp!, ep: ep!, dp: dp! };
}

// ---------------------------------------------------------------------------
// URL ↔ State conversion
// ---------------------------------------------------------------------------

type ParamMap = { key: keyof UrlState; param: string };

// Flat number/string/boolean fields (not layout overrides or calibration).
const FLAT_PARAMS: ParamMap[] = [
  { key: 'modelId', param: 'model' },
  { key: 'gpuId', param: 'gpu' },
  { key: 'numGpus', param: 'n' },
  { key: 'gpusPerNode', param: 'gpn' },
  { key: 'intraId', param: 'intra' },
  { key: 'interId', param: 'inter' },
  { key: 'quant', param: 'quant' },
  { key: 'inputLen', param: 'in' },
  { key: 'outputLen', param: 'out' },
  { key: 'batchSize', param: 'b' },
  { key: 'prefillRatioOn', param: 'pf_on' },
  { key: 'prefillRatio', param: 'pf' },
  { key: 'flashAttention', param: 'fa' },
  { key: 'disaggOn', param: 'pd' },
  { key: 'prefillGpus', param: 'p_n' },
  { key: 'decodeGpus', param: 'd_n' },
  { key: 'prefillDp', param: 'p_dp' },
  { key: 'decodeDp', param: 'd_dp' },
  { key: 'kvOverlap', param: 'kv' },
  { key: 'dp', param: 'dp' },
  { key: 'ep', param: 'ep' },
  { key: 'sdOn', param: 'sd' },
  { key: 'draftModelId', param: 'draft' },
  { key: 'draftTp', param: 'dtp' },
  { key: 'gamma', param: 'gamma' },
  { key: 'acceptanceRate', param: 'ar' },
  { key: 'headroom', param: 'hr' },
];

const CAL_PARAMS: { key: keyof Calibration; param: string }[] = [
  { key: 'mfuPrefill', param: 'mfu' },
  { key: 'bwEffDecode', param: 'bwe' },
  { key: 'commEffIntra', param: 'cei' },
  { key: 'commEffInter', param: 'cex' },
  { key: 'tpCommOverlap', param: 'tpo' },
  { key: 'epCommOverlap', param: 'epo' },
  { key: 'ppCommOverlap', param: 'ppo' },
  { key: 'alphaIntraMs', param: 'ai' },
  { key: 'alphaInterMs', param: 'ax' },
];

const LAYOUT_PARAMS: { key: 'layoutOverride' | 'prefillLayoutOverride' | 'decodeLayoutOverride'; param: string }[] = [
  { key: 'layoutOverride', param: 'lo' },
  { key: 'prefillLayoutOverride', param: 'plo' },
  { key: 'decodeLayoutOverride', param: 'dlo' },
];

function stateToParams(state: UrlState): URLSearchParams {
  const params = new URLSearchParams();

  for (const { key, param } of FLAT_PARAMS) {
    const val = state[key];
    const def = URL_DEFAULTS[key];
    if (val === def) continue; // omit defaults

    if (typeof val === 'boolean') {
      // Only emit booleans that differ from default
      params.set(param, val ? '1' : '0');
    } else if (typeof val === 'number') {
      params.set(param, String(val));
    } else if (typeof val === 'string') {
      if (val !== '') params.set(param, val);
    }
  }

  // Calibration: only emit fields that differ from the calibrated preset
  for (const { key, param } of CAL_PARAMS) {
    const val = state.cal[key];
    const def = CALIBRATED_PRESET[key];
    if (Math.abs(val - def) > 1e-9) {
      params.set(param, String(val));
    }
  }

  // Layout overrides
  for (const { key, param } of LAYOUT_PARAMS) {
    const layout = state[key];
    if (layout) params.set(param, serializeLayout(layout));
  }

  return params;
}

function paramsToState(searchParams: URLSearchParams): Partial<UrlState> {
  const partial: Partial<UrlState> = {};
  const p = (name: string) => searchParams.get(name);
  const num = (name: string) => {
    const v = searchParams.get(name);
    return v !== null ? Number(v) : undefined;
  };
  const bool = (name: string) => {
    const v = searchParams.get(name);
    return v === null ? undefined : v === '1';
  };

  // Flat fields
  for (const { key, param } of FLAT_PARAMS) {
    const raw = p(param);
    if (raw === null) continue;
    const def = URL_DEFAULTS[key];
    if (typeof def === 'boolean') {
      (partial as any)[key] = bool(param);
    } else if (typeof def === 'number') {
      const n = num(param);
      if (n !== undefined && Number.isFinite(n)) (partial as any)[key] = n;
    } else {
      (partial as any)[key] = raw;
    }
  }

  // Calibration (partial: only override fields present in URL)
  const calOverrides: Partial<Calibration> = {};
  for (const { key, param } of CAL_PARAMS) {
    const v = num(param);
    if (v !== undefined && Number.isFinite(v)) calOverrides[key] = v;
  }
  if (Object.keys(calOverrides).length > 0) {
    partial.cal = { ...CALIBRATED_PRESET, ...calOverrides };
  }

  // Layout overrides
  for (const { key, param } of LAYOUT_PARAMS) {
    const raw = p(param);
    if (raw !== null) {
      const layout = parseLayout(raw);
      if (layout) (partial as any)[key] = layout;
    }
  }

  return partial;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

let initialOverrides: Partial<UrlState> | null = null;

export function useUrlState() {
  const [state, setState] = useState<UrlState>(() => {
    if (initialOverrides === null) {
      try {
        initialOverrides = paramsToState(new URLSearchParams(window.location.search));
      } catch {
        initialOverrides = {};
      }
    }
    return { ...URL_DEFAULTS, ...initialOverrides };
  });

  // Sync state → URL (replaceState to avoid polluting browser history).
  // Preserves the `lang` param managed by LocaleProvider.
  useEffect(() => {
    const params = stateToParams(state);
    try {
      const lang = new URLSearchParams(window.location.search).get('lang');
      if (lang) params.set('lang', lang);
    } catch {
      // ignore
    }
    const qs = params.toString();
    const newUrl = qs ? `${window.location.pathname}?${qs}` : window.location.pathname;
    window.history.replaceState(null, '', newUrl);
  }, [state]);

  return [state, setState] as const;
}

// Read the locale preference from URL query params (called by LocaleProvider).
export function readLocaleFromUrl(): string | null {
  try {
    return new URLSearchParams(window.location.search).get('lang');
  } catch {
    return null;
  }
}

// Write locale to URL (called when user changes language).
export function writeLocaleToUrl(locale: string) {
  try {
    const params = new URLSearchParams(window.location.search);
    if (locale === 'en') {
      params.delete('lang');
    } else {
      params.set('lang', locale);
    }
    const qs = params.toString();
    const newUrl = qs ? `${window.location.pathname}?${qs}` : window.location.pathname;
    window.history.replaceState(null, '', newUrl);
  } catch {
    // SSR or unsupported — silently ignore
  }
}
