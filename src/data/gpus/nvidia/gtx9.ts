// NVIDIA gtx9 generation; migrated from tps catalog (scripts/migrate-gpus.mjs).
import type { GpuSpec } from '../../../core/types';

export const GPUS_GTX9: Record<string, GpuSpec> = {
  gtx980ti: {"id":"gtx980ti","name":"GTX 980 Ti","vramGb":6,"bwGbps":336,"peakTflops":{"bf16":6,"int8":6}},
  gtx980: {"id":"gtx980","name":"GTX 980","vramGb":4,"bwGbps":224,"peakTflops":{"bf16":4,"int8":4}},
  gtx970: {"id":"gtx970","name":"GTX 970","vramGb":4,"bwGbps":196,"peakTflops":{"bf16":4,"int8":4}},
  gtx960: {"id":"gtx960","name":"GTX 960","vramGb":2,"bwGbps":112,"peakTflops":{"bf16":2,"int8":2}},
  gtx950: {"id":"gtx950","name":"GTX 950","vramGb":2,"bwGbps":105,"peakTflops":{"bf16":2,"int8":2}},
};
