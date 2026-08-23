// NVIDIA rtx20 generation; migrated from tps catalog (scripts/migrate-gpus.mjs).
import type { GpuSpec } from '../../../core/types';

export const GPUS_RTX20: Record<string, GpuSpec> = {
  rtx2080ti: {"id":"rtx2080ti","name":"RTX 2080 Ti","vramGb":11,"bwGbps":616,"peakTflops":{"bf16":27,"int8":54}},
  rtx2080s: {"id":"rtx2080s","name":"RTX 2080 SUPER","vramGb":8,"bwGbps":496,"peakTflops":{"bf16":22,"int8":44}},
  rtx2080: {"id":"rtx2080","name":"RTX 2080","vramGb":8,"bwGbps":448,"peakTflops":{"bf16":20,"int8":41}},
  rtx2070s: {"id":"rtx2070s","name":"RTX 2070 SUPER","vramGb":8,"bwGbps":448,"peakTflops":{"bf16":18,"int8":37}},
  rtx2070: {"id":"rtx2070","name":"RTX 2070","vramGb":8,"bwGbps":448,"peakTflops":{"bf16":15,"int8":30}},
  rtx2060s: {"id":"rtx2060s","name":"RTX 2060 SUPER","vramGb":8,"bwGbps":448,"peakTflops":{"bf16":14,"int8":28}},
  rtx2060_12g: {"id":"rtx2060_12g","name":"RTX 2060 12GB","vramGb":12,"bwGbps":336,"peakTflops":{"bf16":13,"int8":26}},
  rtx2060: {"id":"rtx2060","name":"RTX 2060","vramGb":6,"bwGbps":336,"peakTflops":{"bf16":13,"int8":26}},
};
