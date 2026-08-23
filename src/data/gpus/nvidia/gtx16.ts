// NVIDIA gtx16 generation; migrated from tps catalog (scripts/migrate-gpus.mjs).
import type { GpuSpec } from '../../../core/types';

export const GPUS_GTX16: Record<string, GpuSpec> = {
  gtx1660ti: {"id":"gtx1660ti","name":"GTX 1660 Ti","vramGb":6,"bwGbps":288,"peakTflops":{"bf16":11,"int8":11}},
  gtx1660s: {"id":"gtx1660s","name":"GTX 1660 SUPER","vramGb":6,"bwGbps":336,"peakTflops":{"bf16":10,"int8":10}},
  gtx1660: {"id":"gtx1660","name":"GTX 1660","vramGb":6,"bwGbps":192,"peakTflops":{"bf16":10,"int8":10}},
  gtx1650s: {"id":"gtx1650s","name":"GTX 1650 SUPER","vramGb":4,"bwGbps":192,"peakTflops":{"bf16":6,"int8":6}},
  gtx1650ti: {"id":"gtx1650ti","name":"GTX 1650 Ti","vramGb":4,"bwGbps":192,"peakTflops":{"bf16":5,"int8":5}},
  gtx1650: {"id":"gtx1650","name":"GTX 1650","vramGb":4,"bwGbps":128,"peakTflops":{"bf16":5,"int8":5}},
  gtx1630: {"id":"gtx1630","name":"GTX 1630","vramGb":4,"bwGbps":96,"peakTflops":{"bf16":3,"int8":3}},
};
