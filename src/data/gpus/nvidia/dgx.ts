// NVIDIA dgx generation; migrated from tps catalog (scripts/migrate-gpus.mjs).
import type { GpuSpec } from '../../../core/types';

export const GPUS_DGX: Record<string, GpuSpec> = {
  dgx_spark: {"id":"dgx_spark","name":"DGX Spark (GB10)","vramGb":128,"bwGbps":273,"peakTflops":{"bf16":200,"int8":400,"int4":1000}},
};
