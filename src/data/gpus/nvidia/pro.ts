// NVIDIA pro generation; migrated from tps catalog (scripts/migrate-gpus.mjs).
import type { GpuSpec } from '../../../core/types';

export const GPUS_PRO: Record<string, GpuSpec> = {
  rtx_pro_6000: {"id":"rtx_pro_6000","name":"RTX PRO 6000","vramGb":96,"bwGbps":1792,"peakTflops":{"bf16":209,"int8":419,"int4":838}},
  rtx6000_ada: {"id":"rtx6000_ada","name":"RTX 6000 Ada","vramGb":48,"bwGbps":960,"peakTflops":{"bf16":91,"int8":182,"int4":364}},
  rtx5880_ada: {"id":"rtx5880_ada","name":"RTX 5880 Ada","vramGb":48,"bwGbps":864,"peakTflops":{"bf16":82,"int8":164,"int4":328}},
  rtx5000_ada: {"id":"rtx5000_ada","name":"RTX 5000 Ada","vramGb":32,"bwGbps":576,"peakTflops":{"bf16":66,"int8":133,"int4":266}},
  rtx4500_ada: {"id":"rtx4500_ada","name":"RTX 4500 Ada","vramGb":24,"bwGbps":432,"peakTflops":{"bf16":48,"int8":97,"int4":194}},
  rtx4000_sff_ada: {"id":"rtx4000_sff_ada","name":"RTX 4000 SFF Ada","vramGb":20,"bwGbps":272,"peakTflops":{"bf16":20,"int8":40,"int4":80}},
  rtx4000_ada: {"id":"rtx4000_ada","name":"RTX 4000 Ada","vramGb":20,"bwGbps":360,"peakTflops":{"bf16":26,"int8":52,"int4":104}},
  rtx3500_ada: {"id":"rtx3500_ada","name":"RTX 3500 Ada","vramGb":12,"bwGbps":288,"peakTflops":{"bf16":18,"int8":36,"int4":72}},
  rtx2000_ada: {"id":"rtx2000_ada","name":"RTX 2000 Ada","vramGb":16,"bwGbps":224,"peakTflops":{"bf16":12,"int8":24,"int4":48}},
  rtx_a6000: {"id":"rtx_a6000","name":"RTX A6000","vramGb":48,"bwGbps":768,"peakTflops":{"bf16":77,"int8":154,"int4":309}},
  rtx_a5500: {"id":"rtx_a5500","name":"RTX A5500","vramGb":24,"bwGbps":768,"peakTflops":{"bf16":46,"int8":92,"int4":184}},
  rtx_a5000: {"id":"rtx_a5000","name":"RTX A5000","vramGb":24,"bwGbps":768,"peakTflops":{"bf16":43,"int8":87,"int4":174}},
  rtx_a4500: {"id":"rtx_a4500","name":"RTX A4500","vramGb":20,"bwGbps":640,"peakTflops":{"bf16":31,"int8":62,"int4":125}},
  rtx_a4000: {"id":"rtx_a4000","name":"RTX A4000","vramGb":16,"bwGbps":448,"peakTflops":{"bf16":20,"int8":40,"int4":80}},
  rtx_a2000: {"id":"rtx_a2000","name":"RTX A2000","vramGb":6,"bwGbps":192,"peakTflops":{"bf16":8,"int8":16,"int4":32}},
  rtx_a2000_8g: {"id":"rtx_a2000_8g","name":"RTX A2000 8GB","vramGb":8,"bwGbps":192,"peakTflops":{"bf16":8,"int8":16,"int4":32}},
};
