// NVIDIA datacenter generation; migrated from tps catalog (scripts/migrate-gpus.mjs).
import type { GpuSpec } from '../../../core/types';

export const GPUS_DATACENTER: Record<string, GpuSpec> = {
  // Blackwell
  gb200_nvl72_node: { "id": "gb200_nvl72_node", "name": "GB200 NVL72 (Single Node 2-GPU)", "vramGb": 384, "bwGbps": 16000, "peakTflops": { "bf16": 5000, "fp8": 10000, "int8": 10000, "int4": 20000, "fp4": 20000 }, "nvlinkBwGbps": 3600 },
  b300_sxm: { "id": "b300_sxm", "name": "B300 SXM", "vramGb": 288, "bwGbps": 13000, "peakTflops": { "bf16": 3750, "fp8": 7500, "int8": 7500, "int4": 15000, "fp4": 15000 }, "nvlinkBwGbps": 1800 },
  b200_sxm: { "id": "b200_sxm", "name": "B200 SXM", "vramGb": 180, "bwGbps": 7800, "peakTflops": { "bf16": 2250, "fp8": 4500, "int8": 4500, "int4": 9000, "fp4": 9000 }, "nvlinkBwGbps": 1800 },
  b200_pcie: { "id": "b200_pcie", "name": "B200 PCIe", "vramGb": 180, "bwGbps": 7800, "peakTflops": { "bf16": 1750, "fp8": 3500, "int8": 3500, "int4": 7000, "fp4": 7000 } },
  b100_pcie: { "id": "b100_pcie", "name": "B100 PCIe", "vramGb": 192, "bwGbps": 8000, "peakTflops": { "bf16": 1750, "fp8": 3500, "int8": 3500, "int4": 7000, "fp4": 7000 } },

  // Hopper
  h200_sxm: { "id": "h200_sxm", "name": "H200 SXM", "vramGb": 141, "bwGbps": 4800, "peakTflops": { "bf16": 989, "fp8": 1979, "int8": 1979, "int4": 3958 }, "nvlinkBwGbps": 900 },
  h200_nvl: { "id": "h200_nvl", "name": "H200 NVL", "vramGb": 141, "bwGbps": 4800, "peakTflops": { "bf16": 989, "fp8": 1979, "int8": 1979, "int4": 3958 }, "nvlinkBwGbps": 900 },
  h100_sxm: { "id": "h100_sxm", "name": "H100 SXM5", "vramGb": 80, "bwGbps": 3350, "peakTflops": { "bf16": 989, "fp8": 1979, "int8": 1979, "int4": 3958 }, "nvlinkBwGbps": 900 },
  h100_nvl: { "id": "h100_nvl", "name": "H100 NVL (Dual-GPU Card)", "vramGb": 188, "bwGbps": 7800, "peakTflops": { "bf16": 1978, "fp8": 3958, "int8": 3958, "int4": 7916 }, "nvlinkBwGbps": 900 },
  h100_pcie: { "id": "h100_pcie", "name": "H100 PCIe", "vramGb": 80, "bwGbps": 2000, "peakTflops": { "bf16": 756, "fp8": 1513, "int8": 1513, "int4": 3026 } },
  h800_sxm: { "id": "h800_sxm", "name": "H800 SXM", "vramGb": 80, "bwGbps": 3350, "peakTflops": { "bf16": 989, "fp8": 1979, "int8": 1979, "int4": 3958 }, "nvlinkBwGbps": 400 },
  h800_pcie: { "id": "h800_pcie", "name": "H800 PCIe", "vramGb": 80, "bwGbps": 2000, "peakTflops": { "bf16": 756, "fp8": 1513, "int8": 1513, "int4": 3026 }, "nvlinkBwGbps": 400 },
  h20: { "id": "h20", "name": "H20 SXM", "vramGb": 96, "bwGbps": 4000, "peakTflops": { "bf16": 148, "fp8": 296, "int8": 296, "int4": 592 }, "nvlinkBwGbps": 900 },

  // Ada Lovelace 
  l40s: { "id": "l40s", "name": "L40S", "vramGb": 48, "bwGbps": 864, "peakTflops": { "bf16": 362, "fp8": 724, "int8": 724, "int4": 1448 } },
  l40: { "id": "l40", "name": "L40", "vramGb": 48, "bwGbps": 864, "peakTflops": { "bf16": 181, "fp8": 362, "int8": 362, "int4": 724 } },
  l20: { "id": "l20", "name": "L20", "vramGb": 48, "bwGbps": 864, "peakTflops": { "bf16": 119, "fp8": 239, "int8": 239, "int4": 478 } },
  l4: { "id": "l4", "name": "L4", "vramGb": 24, "bwGbps": 300, "peakTflops": { "bf16": 121, "fp8": 242, "int8": 242, "int4": 484 } },

  // Ampere
  a100_sxm_80g: { "id": "a100_sxm_80g", "name": "A100 SXM4 80G", "vramGb": 80, "bwGbps": 2039, "peakTflops": { "bf16": 312, "int8": 624, "int4": 1248 }, "nvlinkBwGbps": 600 },
  a100_sxm_40g: { "id": "a100_sxm_40g", "name": "A100 SXM4 40G", "vramGb": 40, "bwGbps": 1555, "peakTflops": { "bf16": 312, "int8": 624, "int4": 1248 }, "nvlinkBwGbps": 600 },
  a100_pcie_80g: { "id": "a100_pcie_80g", "name": "A100 PCIe 80G", "vramGb": 80, "bwGbps": 1935, "peakTflops": { "bf16": 312, "int8": 624, "int4": 1248 }, "nvlinkBwGbps": 600 },
  a100_pcie_40g: { "id": "a100_pcie_40g", "name": "A100 PCIe 40G", "vramGb": 40, "bwGbps": 1555, "peakTflops": { "bf16": 312, "int8": 624, "int4": 1248 }, "nvlinkBwGbps": 600 },
  a800_sxm_80g: { "id": "a800_sxm_80g", "name": "A800 SXM 80G", "vramGb": 80, "bwGbps": 2039, "peakTflops": { "bf16": 312, "int8": 624, "int4": 1248 }, "nvlinkBwGbps": 400 },
  a800_pcie_80g: { "id": "a800_pcie_80g", "name": "A800 PCIe 80G", "vramGb": 80, "bwGbps": 1935, "peakTflops": { "bf16": 312, "int8": 624, "int4": 1248 }, "nvlinkBwGbps": 400 },
  a40: { "id": "a40", "name": "A40", "vramGb": 48, "bwGbps": 696, "peakTflops": { "bf16": 149, "int8": 300, "int4": 600 } },
  a30: { "id": "a30", "name": "A30", "vramGb": 24, "bwGbps": 933, "peakTflops": { "bf16": 165, "int8": 330, "int4": 661 }, "nvlinkBwGbps": 200 },
  a10: { "id": "a10", "name": "A10", "vramGb": 24, "bwGbps": 600, "peakTflops": { "bf16": 125, "int8": 250, "int4": 500 } },
  a16: { "id": "a16", "name": "A16", "vramGb": 64, "bwGbps": 800, "peakTflops": { "bf16": 184, "int8": 368, "int4": 736 } },

  // Volta / Turing
  v100_sxm2_32g: { "id": "v100_sxm2_32g", "name": "V100 SXM2 32G", "vramGb": 32, "bwGbps": 900, "peakTflops": { "bf16": 125 }, "nvlinkBwGbps": 300 },
  v100_sxm2_16g: { "id": "v100_sxm2_16g", "name": "V100 SXM2 16G", "vramGb": 16, "bwGbps": 900, "peakTflops": { "bf16": 125 }, "nvlinkBwGbps": 300 },
  v100_pcie_32g: { "id": "v100_pcie_32g", "name": "V100 PCIe 32G", "vramGb": 32, "bwGbps": 900, "peakTflops": { "bf16": 125 } },
  t4: { "id": "t4", "name": "Tesla T4", "vramGb": 16, "bwGbps": 300, "peakTflops": { "bf16": 65, "int8": 130 } },
};
