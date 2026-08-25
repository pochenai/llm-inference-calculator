// Aggregated model catalog: all migrated models, tiered by parameter count.
import type { ModelSpec } from '../../core/types';
import { MODELS_6B } from './6B';
import { MODELS_24B } from './24B';
import { MODELS_32B } from './32B';
import { MODELS_128B } from './128B';
import { MODELS_500B } from './500B';
import { MODELS_5000B } from './5000B';

export const ALL_MODELS: Record<string, ModelSpec> = {
  ...MODELS_6B,
  ...MODELS_24B,
  ...MODELS_32B,
  ...MODELS_128B,
  ...MODELS_500B,
  ...MODELS_5000B,
};

export function model(id: string): ModelSpec {
  const m = ALL_MODELS[id];
  if (!m) throw new Error(`unknown model id: ${id}`);
  return m;
}

// Size tier labels for UI filtering. Each tier key corresponds to a model
// data file (6B.ts, 24B.ts, etc.).
export const SIZE_TIERS = [
  { value: '6B', label: '≤6B' },
  { value: '24B', label: '≤24B' },
  { value: '32B', label: '≤32B' },
  { value: '128B', label: '≤128B' },
  { value: '500B', label: '≤500B' },
  { value: '5000B', label: '>500B' },
] as const;

// Build a model-id → tier-value map from the per-tier record keys.
// This avoids adding a field to every model entry.
function buildTierLookup(
  tiers: { value: string; ids: string[] }[],
): Map<string, string> {
  const map = new Map<string, string>();
  for (const t of tiers) for (const id of t.ids) map.set(id, t.value);
  return map;
}

export const MODEL_SIZE_TIER: Map<string, string> = buildTierLookup([
  { value: '6B', ids: Object.keys(MODELS_6B) },
  { value: '24B', ids: Object.keys(MODELS_24B) },
  { value: '32B', ids: Object.keys(MODELS_32B) },
  { value: '128B', ids: Object.keys(MODELS_128B) },
  { value: '500B', ids: Object.keys(MODELS_500B) },
  { value: '5000B', ids: Object.keys(MODELS_5000B) },
]);
