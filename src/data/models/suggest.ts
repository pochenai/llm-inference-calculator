// Draft model suggestion utilities for speculative decoding.
// Strategy: prefer same family, pick the largest candidate in the target size range.

import type { ModelSpec } from '../../core/types';

// Extract a family prefix from model id.
// Examples:
//   "llama3_1_70b" -> "llama"
//   "qwen3_235b" -> "qwen"
//   "gemma3_27b" -> "gemma"
//   "granite_4_1_30b" -> "granite"
//   "nemotron_3_nano" -> "nemotron"
export function familyOf(model: ModelSpec): string {
  const id = model.id.toLowerCase();
  const match = id.match(/^([a-z][a-z0-9]*?)(?=[\d_]|$)/);
  if (match && match[1]) return match[1];
  const parts = id.split(/[_\d]/);
  return parts[0] || id;
}

// Return all dense models whose paramsB falls within [paramsB * ratioMin, paramsB * ratioMax].
// Sorted by paramsB descending (largest first).
export function modelsInRange(
  allModels: Record<string, ModelSpec>,
  paramsB: number,
  ratioMin: number,
  ratioMax: number,
): ModelSpec[] {
  const minP = paramsB * ratioMin;
  const maxP = paramsB * ratioMax;
  return Object.values(allModels)
    .filter(m => m.paramsB >= minP && m.paramsB <= maxP && m.type === 'dense')
    .sort((a, b) => b.paramsB - a.paramsB);
}

// Suggest a draft model for the given main model.
// Default ratio: 0.1 (10x smaller) to 0.2 (5x smaller).
// Strategy: prefer same family (higher expected acceptance rate), otherwise pick largest overall.
export function suggestDraftModel(
  main: ModelSpec,
  allModels: Record<string, ModelSpec>,
  ratioMin: number = 0.1,
  ratioMax: number = 0.2,
): ModelSpec | null {
  // Never suggest the model itself or something larger
  const candidates = modelsInRange(allModels, main.paramsB, ratioMin, ratioMax).filter(
    m => m.id !== main.id,
  );

  if (candidates.length === 0) return null;

  const mainFamily = familyOf(main);

  // Tier 1: same family (highest expected acceptance rate)
  const sameFamily = candidates.filter(m => familyOf(m) === mainFamily);
  if (sameFamily.length > 0) return sameFamily[0]!; // already sorted desc by paramsB

  // Tier 2: largest overall candidate (better acceptance rate than smaller models)
  return candidates[0]!;
}
