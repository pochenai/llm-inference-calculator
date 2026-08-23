// GPU spec catalog used by the benchmark comparison. English comments.
// Values are the datasheet / commonly-cited numbers; re-verify before relying on them.

import type { GpuSpec } from '../core/types.js';

export const GPUS: Record<string, GpuSpec> = {
  h100_sxm: {
    id: 'h100_sxm',
    name: 'H100 SXM5',
    vramGb: 80,
    bwGbps: 3350,
    peakTflops: { bf16: 989, fp8: 1979, int8: 1979, int4: 3958 },
    nvlinkBwGbps: 900,
  },
  h100_pcie: {
    id: 'h100_pcie',
    name: 'H100 PCIe',
    vramGb: 80,
    bwGbps: 2000,
    peakTflops: { bf16: 756, fp8: 1513, int8: 1513, int4: 3026 },
  },
  a100_sxm: {
    id: 'a100_sxm',
    name: 'A100 SXM4 80G',
    vramGb: 80,
    bwGbps: 2000,
    peakTflops: { bf16: 312, int8: 624, int4: 1248 },
    nvlinkBwGbps: 600,
  },
  a100_pcie: {
    id: 'a100_pcie',
    name: 'A100 PCIe 80G',
    vramGb: 80,
    bwGbps: 2000,
    peakTflops: { bf16: 312, int8: 624, int4: 1248 },
  },
  h200_sxm: {
    id: 'h200_sxm',
    name: 'H200 SXM',
    vramGb: 141,
    bwGbps: 4800,
    peakTflops: { bf16: 989, fp8: 1979, int8: 1979, int4: 3958 },
    nvlinkBwGbps: 900,
  },
  h20: {
    id: 'h20',
    name: 'H20 96G',
    vramGb: 96,
    bwGbps: 4000,
    peakTflops: { bf16: 148, fp8: 296, int8: 296, int4: 592 },
    nvlinkBwGbps: 900,
  },
};

export function gpu(id: string): GpuSpec {
  const g = GPUS[id];
  if (!g) throw new Error(`unknown gpu id: ${id}`);
  return g;
}
