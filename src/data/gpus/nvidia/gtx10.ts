// NVIDIA gtx10 generation; migrated from tps catalog (scripts/migrate-gpus.mjs).
import type { GpuSpec } from '../../../core/types';

export const GPUS_GTX10: Record<string, GpuSpec> = {
  gtx1080ti: {"id":"gtx1080ti","name":"GTX 1080 Ti","vramGb":11,"bwGbps":484,"peakTflops":{"bf16":11,"int8":11}},
  gtx1080: {"id":"gtx1080","name":"GTX 1080","vramGb":8,"bwGbps":320,"peakTflops":{"bf16":9,"int8":9}},
  gtx1070ti: {"id":"gtx1070ti","name":"GTX 1070 Ti","vramGb":8,"bwGbps":256,"peakTflops":{"bf16":8,"int8":8}},
  gtx1070: {"id":"gtx1070","name":"GTX 1070","vramGb":8,"bwGbps":256,"peakTflops":{"bf16":7,"int8":7}},
  gtx1060_6g: {"id":"gtx1060_6g","name":"GTX 1060 6GB","vramGb":6,"bwGbps":192,"peakTflops":{"bf16":4,"int8":4}},
  gtx1060_3g: {"id":"gtx1060_3g","name":"GTX 1060 3GB","vramGb":3,"bwGbps":192,"peakTflops":{"bf16":4,"int8":4}},
  gtx1050ti: {"id":"gtx1050ti","name":"GTX 1050 Ti","vramGb":4,"bwGbps":112,"peakTflops":{"bf16":2,"int8":2}},
  gtx1050: {"id":"gtx1050","name":"GTX 1050","vramGb":2,"bwGbps":112,"peakTflops":{"bf16":2,"int8":2}},
};
