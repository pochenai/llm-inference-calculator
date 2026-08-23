// NVIDIA rtx50 generation; migrated from tps catalog (scripts/migrate-gpus.mjs).
import type { GpuSpec } from '../../../core/types';

export const GPUS_RTX50: Record<string, GpuSpec> = {
  rtx5090: {"id":"rtx5090","name":"RTX 5090","vramGb":32,"bwGbps":1792,"peakTflops":{"bf16":209,"int8":419,"int4":838}},
  rtx5080: {"id":"rtx5080","name":"RTX 5080","vramGb":16,"bwGbps":960,"peakTflops":{"bf16":113,"int8":226,"int4":452}},
  rtx5070ti: {"id":"rtx5070ti","name":"RTX 5070 Ti","vramGb":16,"bwGbps":896,"peakTflops":{"bf16":88,"int8":176,"int4":352}},
  rtx5070: {"id":"rtx5070","name":"RTX 5070","vramGb":12,"bwGbps":672,"peakTflops":{"bf16":61,"int8":122,"int4":244}},
  rtx5060ti_16g: {"id":"rtx5060ti_16g","name":"RTX 5060 Ti 16GB","vramGb":16,"bwGbps":448,"peakTflops":{"bf16":24,"int8":48,"int4":96}},
  rtx5060ti: {"id":"rtx5060ti","name":"RTX 5060 Ti","vramGb":8,"bwGbps":448,"peakTflops":{"bf16":24,"int8":48,"int4":96}},
  rtx5060: {"id":"rtx5060","name":"RTX 5060","vramGb":8,"bwGbps":336,"peakTflops":{"bf16":18,"int8":36,"int4":72}},
  rtx5050: {"id":"rtx5050","name":"RTX 5050","vramGb":8,"bwGbps":256,"peakTflops":{"bf16":13,"int8":26,"int4":52}},
  rtx5090_laptop: {"id":"rtx5090_laptop","name":"RTX 5090 Laptop","vramGb":24,"bwGbps":896,"peakTflops":{"bf16":138,"int8":276,"int4":552}},
  rtx5080_laptop: {"id":"rtx5080_laptop","name":"RTX 5080 Laptop","vramGb":16,"bwGbps":576,"peakTflops":{"bf16":80,"int8":160,"int4":320}},
  rtx5070ti_laptop: {"id":"rtx5070ti_laptop","name":"RTX 5070 Ti Laptop","vramGb":12,"bwGbps":448,"peakTflops":{"bf16":56,"int8":112,"int4":224}},
  rtx5070_laptop: {"id":"rtx5070_laptop","name":"RTX 5070 Laptop","vramGb":8,"bwGbps":352,"peakTflops":{"bf16":40,"int8":80,"int4":160}},
  rtx5060_laptop: {"id":"rtx5060_laptop","name":"RTX 5060 Laptop","vramGb":8,"bwGbps":384,"peakTflops":{"bf16":17,"int8":33,"int4":66}},
  rtx5050_laptop: {"id":"rtx5050_laptop","name":"RTX 5050 Laptop","vramGb":8,"bwGbps":224,"peakTflops":{"bf16":13,"int8":26,"int4":52}},
};
