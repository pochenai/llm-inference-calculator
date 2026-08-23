// Aggregated NVIDIA GPU catalog, one file per generation.
import type { GpuSpec } from '../../../core/types';
import { GPUS_DATACENTER } from './datacenter';
import { GPUS_DGX } from './dgx';
import { GPUS_GTX9 } from './gtx9';
import { GPUS_GTX10 } from './gtx10';
import { GPUS_GTX16 } from './gtx16';
import { GPUS_PRO } from './pro';
import { GPUS_RTX20 } from './rtx20';
import { GPUS_RTX30 } from './rtx30';
import { GPUS_RTX40 } from './rtx40';
import { GPUS_RTX50 } from './rtx50';

export const ALL_GPUS: Record<string, GpuSpec> = {
  ...GPUS_DATACENTER,
  ...GPUS_DGX,
  ...GPUS_GTX9,
  ...GPUS_GTX10,
  ...GPUS_GTX16,
  ...GPUS_PRO,
  ...GPUS_RTX20,
  ...GPUS_RTX30,
  ...GPUS_RTX40,
  ...GPUS_RTX50,
};

export function gpu(id: string): GpuSpec {
  const g = ALL_GPUS[id];
  if (!g) throw new Error(`unknown gpu id: ${id}`);
  return g;
}
