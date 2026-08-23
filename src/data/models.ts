// Model spec catalog used by the benchmark comparison. English comments.
// Architectures taken from public config.json files; re-verify before relying on them.

import type { ModelSpec } from '../core/types.js';

export const MODELS: Record<string, ModelSpec> = {
  llama2_70b: {
    id: 'llama2_70b',
    name: 'Llama 2 70B',
    type: 'dense',
    paramsB: 70,
    layers: 80,
    hiddenSize: 8192,
    kvHeads: 8,
    headDim: 128,
    maxCtx: 4096,
  },
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
  },
  llama3_1_8b: {
    id: 'llama3_1_8b',
    name: 'Llama 3.1 8B',
    type: 'dense',
    paramsB: 8.03,
    layers: 32,
    hiddenSize: 4096,
    kvHeads: 8,
    headDim: 128,
    maxCtx: 131072,
  },
  llama3_3_70b: {
    id: 'llama3_3_70b',
    name: 'Llama 3.3 70B',
    type: 'dense',
    paramsB: 70.6,
    layers: 80,
    hiddenSize: 8192,
    kvHeads: 8,
    headDim: 128,
    maxCtx: 131072,
  },
  llama3_1_405b: {
    id: 'llama3_1_405b',
    name: 'Llama 3.1 405B',
    type: 'dense',
    paramsB: 405,
    layers: 126,
    hiddenSize: 16384,
    kvHeads: 8,
    headDim: 128,
    maxCtx: 131072,
  },
  qwen3_235b: {
    id: 'qwen3_235b',
    name: 'Qwen3 235B A22B',
    type: 'moe',
    paramsB: 235,
    layers: 94,
    hiddenSize: 4096,
    kvHeads: 4,
    headDim: 128,
    maxCtx: 131072,
    moe: { experts: 128, expertsPerToken: 8, activeParamsB: 22, execution: 'shared_routed' },
  },
  deepseek_v3: {
    id: 'deepseek_v3',
    name: 'DeepSeek V3 / V3.1',
    type: 'moe',
    paramsB: 671,
    layers: 61,
    hiddenSize: 7168,
    kvHeads: 128,
    headDim: 128,
    maxCtx: 131072,
    mlaRatio: 0.0176,
    moe: { experts: 256, expertsPerToken: 8, activeParamsB: 37, execution: 'shared_routed' },
  },
};

export function model(id: string): ModelSpec {
  const m = MODELS[id];
  if (!m) throw new Error(`unknown model id: ${id}`);
  return m;
}
