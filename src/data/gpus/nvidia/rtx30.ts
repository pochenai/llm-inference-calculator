// NVIDIA rtx30 generation; migrated from tps catalog (scripts/migrate-gpus.mjs).
import type { GpuSpec } from '../../../core/types';

export const GPUS_RTX30: Record<string, GpuSpec> = {
  rtx3090ti: {"id":"rtx3090ti","name":"RTX 3090 Ti","vramGb":24,"bwGbps":1008,"peakTflops":{"bf16":80,"int8":160,"int4":320}},
  rtx3090: {"id":"rtx3090","name":"RTX 3090","vramGb":24,"bwGbps":936,"peakTflops":{"bf16":71,"int8":142,"int4":284}},
  rtx3080ti: {"id":"rtx3080ti","name":"RTX 3080 Ti","vramGb":12,"bwGbps":912,"peakTflops":{"bf16":68,"int8":136,"int4":272}},
  rtx3080_12g: {"id":"rtx3080_12g","name":"RTX 3080 12GB","vramGb":12,"bwGbps":912,"peakTflops":{"bf16":68,"int8":136,"int4":272}},
  rtx3080: {"id":"rtx3080","name":"RTX 3080","vramGb":10,"bwGbps":760,"peakTflops":{"bf16":60,"int8":119,"int4":238}},
  rtx3070ti: {"id":"rtx3070ti","name":"RTX 3070 Ti","vramGb":8,"bwGbps":608,"peakTflops":{"bf16":43,"int8":87,"int4":174}},
  rtx3070: {"id":"rtx3070","name":"RTX 3070","vramGb":8,"bwGbps":448,"peakTflops":{"bf16":40,"int8":80,"int4":160}},
  rtx3060ti: {"id":"rtx3060ti","name":"RTX 3060 Ti","vramGb":8,"bwGbps":448,"peakTflops":{"bf16":32,"int8":65,"int4":130}},
  rtx3060: {"id":"rtx3060","name":"RTX 3060","vramGb":12,"bwGbps":360,"peakTflops":{"bf16":25,"int8":51,"int4":102}},
  rtx3050: {"id":"rtx3050","name":"RTX 3050","vramGb":8,"bwGbps":224,"peakTflops":{"bf16":18,"int8":36,"int4":72}},
  rtx3050_6g: {"id":"rtx3050_6g","name":"RTX 3050 6GB","vramGb":6,"bwGbps":168,"peakTflops":{"bf16":14,"int8":28,"int4":56}},
  rtx3080ti_laptop: {"id":"rtx3080ti_laptop","name":"RTX 3080 Ti Laptop","vramGb":16,"bwGbps":512,"peakTflops":{"bf16":46,"int8":92,"int4":184}},
  rtx3080_laptop: {"id":"rtx3080_laptop","name":"RTX 3080 Laptop","vramGb":16,"bwGbps":512,"peakTflops":{"bf16":46,"int8":92,"int4":184}},
  rtx3070ti_laptop: {"id":"rtx3070ti_laptop","name":"RTX 3070 Ti Laptop","vramGb":8,"bwGbps":384,"peakTflops":{"bf16":37,"int8":74,"int4":148}},
  rtx3070_laptop: {"id":"rtx3070_laptop","name":"RTX 3070 Laptop","vramGb":8,"bwGbps":384,"peakTflops":{"bf16":29,"int8":58,"int4":116}},
  rtx3060_laptop: {"id":"rtx3060_laptop","name":"RTX 3060 Laptop","vramGb":6,"bwGbps":336,"peakTflops":{"bf16":20,"int8":40,"int4":80}},
  rtx3050ti_laptop: {"id":"rtx3050ti_laptop","name":"RTX 3050 Ti Laptop","vramGb":4,"bwGbps":192,"peakTflops":{"bf16":11,"int8":22,"int4":44}},
  rtx3050_laptop: {"id":"rtx3050_laptop","name":"RTX 3050 Laptop","vramGb":4,"bwGbps":192,"peakTflops":{"bf16":9,"int8":18,"int4":36}},
};
