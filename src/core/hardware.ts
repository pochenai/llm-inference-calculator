// Hardware specs and interconnect resolution.

import type { GpuSpec, Interconnect, QuantPrecision } from './types';
import { CalcError } from './errors';

// PCIe fallback when a GPU has no NVLink. Conservative Gen4 x16 figure.
export const PCIE_BW_GBPS = 32;

// intraNodeBwGbpsOverride lets callers (e.g. the UI) pin an explicit
// intra-node fabric instead of the GPU-datasheet default.
export function resolveInterconnect(
  gpu: GpuSpec,
  interNodeBwGbps: number,
  intraNodeBwGbpsOverride?: number,
): Interconnect {
  return {
    intraNodeBwGbps: intraNodeBwGbpsOverride ?? gpu.nvlinkBwGbps ?? PCIE_BW_GBPS,
    interNodeBwGbps,
  };
}

// Precisions considered equivalent when a GPU datasheet lacks one column
// (e.g. fp16 vs bf16 peak are the same silicon limit).
const EQUIV_GROUPS: QuantPrecision[][] = [
  ['fp16', 'bf16'],
  ['fp8', 'int8'],
  ['int4', 'fp4'],
];

// Whether a GPU has a usable peak-FLOPs entry for the given precision
// (directly or via an equivalence-group fallback).
export function gpuSupportsQuant(gpu: GpuSpec, quant: QuantPrecision): boolean {
  if (gpu.peakTflops[quant] != null) return true;
  const group = EQUIV_GROUPS.find((g) => g.includes(quant));
  if (group) {
    for (const alt of group) {
      if (gpu.peakTflops[alt] != null) return true;
    }
  }
  return false;
}

// Dense peak FLOPs for the requested precision, in FLOPs/s.
// Throws when the datasheet has no usable column, to avoid silent mis-modeling.
export function peakFlopsOf(gpu: GpuSpec, quant: QuantPrecision): number {
  const direct = gpu.peakTflops[quant];
  if (direct != null) return direct * 1e12;
  const group = EQUIV_GROUPS.find((g) => g.includes(quant));
  if (group) {
    for (const alt of group) {
      const v = gpu.peakTflops[alt];
      if (v != null) return v * 1e12;
    }
  }
  throw new CalcError(
    'missing-peak-flops',
    `GPU ${gpu.id} has no peak FLOPs column for precision '${quant}' (or an equivalent)`,
  );
}
