// GPU specs for the calibration entries. Re-uses the migrated NVIDIA catalog
// (src/data/gpus/nvidia) and adds GPUs missing from it (H20 is not in tps).
import type { GpuSpec } from '../../core/types';
import { ALL_GPUS } from '../gpus/nvidia';

const EXTRA: Record<string, GpuSpec> = {
  h20: {
    id: 'h20',
    name: 'H20 96G',
    vramGb: 96,
    bwGbps: 4000,
    peakTflops: { bf16: 148, fp8: 296, int8: 296, int4: 592 },
    nvlinkBwGbps: 900,
  },
};

export const GPUS: Record<string, GpuSpec> = { ...ALL_GPUS, ...EXTRA };

export function gpu(id: string): GpuSpec {
  const g = GPUS[id];
  if (!g) throw new Error(`unknown gpu id: ${id}`);
  return g;
}
