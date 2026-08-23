// NVIDIA rtx40 generation; migrated from tps catalog (scripts/migrate-gpus.mjs).
import type { GpuSpec } from '../../../core/types';

export const GPUS_RTX40: Record<string, GpuSpec> = {
  rtx4090: {"id":"rtx4090","name":"RTX 4090","vramGb":24,"bwGbps":1008,"peakTflops":{"bf16":165,"int8":330,"int4":661}},
  rtx4080s: {"id":"rtx4080s","name":"RTX 4080 SUPER","vramGb":16,"bwGbps":736,"peakTflops":{"bf16":104,"int8":208,"int4":416}},
  rtx4080: {"id":"rtx4080","name":"RTX 4080","vramGb":16,"bwGbps":717,"peakTflops":{"bf16":97,"int8":194,"int4":388}},
  rtx4070tis: {"id":"rtx4070tis","name":"RTX 4070 Ti SUPER","vramGb":16,"bwGbps":672,"peakTflops":{"bf16":88,"int8":176,"int4":352}},
  rtx4070ti: {"id":"rtx4070ti","name":"RTX 4070 Ti","vramGb":12,"bwGbps":672,"peakTflops":{"bf16":80,"int8":160,"int4":320}},
  rtx4070s: {"id":"rtx4070s","name":"RTX 4070 SUPER","vramGb":12,"bwGbps":504,"peakTflops":{"bf16":71,"int8":142,"int4":284}},
  rtx4070: {"id":"rtx4070","name":"RTX 4070","vramGb":12,"bwGbps":504,"peakTflops":{"bf16":58,"int8":116,"int4":233}},
  rtx4060ti_16g: {"id":"rtx4060ti_16g","name":"RTX 4060 Ti 16GB","vramGb":16,"bwGbps":288,"peakTflops":{"bf16":45,"int8":90,"int4":181}},
  rtx4060ti: {"id":"rtx4060ti","name":"RTX 4060 Ti","vramGb":8,"bwGbps":288,"peakTflops":{"bf16":45,"int8":90,"int4":181}},
  rtx4060: {"id":"rtx4060","name":"RTX 4060","vramGb":8,"bwGbps":272,"peakTflops":{"bf16":30,"int8":60,"int4":121}},
  rtx4090_laptop: {"id":"rtx4090_laptop","name":"RTX 4090 Laptop","vramGb":16,"bwGbps":576,"peakTflops":{"bf16":82,"int8":165,"int4":330}},
  rtx4080_laptop: {"id":"rtx4080_laptop","name":"RTX 4080 Laptop","vramGb":12,"bwGbps":432,"peakTflops":{"bf16":58,"int8":116,"int4":233}},
  rtx4070_laptop: {"id":"rtx4070_laptop","name":"RTX 4070 Laptop","vramGb":8,"bwGbps":336,"peakTflops":{"bf16":40,"int8":80,"int4":161}},
  rtx4060_laptop: {"id":"rtx4060_laptop","name":"RTX 4060 Laptop","vramGb":8,"bwGbps":272,"peakTflops":{"bf16":30,"int8":60,"int4":121}},
  rtx4050_laptop: {"id":"rtx4050_laptop","name":"RTX 4050 Laptop","vramGb":6,"bwGbps":192,"peakTflops":{"bf16":19,"int8":38,"int4":76}},
};
