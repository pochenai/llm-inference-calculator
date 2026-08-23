// Model specs for the calibration entries. Re-uses the migrated catalog
// (src/data/models) and adds models missing from it (GPT-J is not in tps).
import type { ModelSpec } from '../../core/types';
import { ALL_MODELS } from '../models';

const EXTRA: Record<string, ModelSpec> = {
  gptj_6b: {
    id: 'gptj_6b',
    name: 'GPT-J 6B',
    type: 'dense',
    paramsB: 6.05,
    layers: 28,
    hiddenSize: 4096,
    kvHeads: 16,
    headDim: 256,
    maxCtx: 2048,
    hf: 'https://huggingface.co/EleutherAI/gpt-j-6b',
  },
};

export const MODELS: Record<string, ModelSpec> = { ...ALL_MODELS, ...EXTRA };

export function model(id: string): ModelSpec {
  const m = MODELS[id];
  if (!m) throw new Error(`unknown model id: ${id}`);
  return m;
}
