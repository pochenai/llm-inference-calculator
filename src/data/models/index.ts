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
