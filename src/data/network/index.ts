// Intra-node Connections (节点内)
export const INTRA_NODES_CONNECTION = [
    // NVLink / NVSwitch (GPU-to-GPU / NVLink Network Switch)
    { id: 'nvlink5', label: 'NVLink 5.0 (Blackwell)', bw: 1800, scope: 'intra' }, // 1.8 TB/s bidirectional
    { id: 'nvlink4', label: 'NVLink 4.0 (Hopper)', bw: 900, scope: 'intra' },     // 900 GB/s bidirectional
    { id: 'nvlink3', label: 'NVLink 3.0 (Ampere)', bw: 600, scope: 'intra' },     // 600 GB/s bidirectional
    { id: 'nvswitch', label: 'NVSwitch Board', bw: 900, scope: 'intra' },

    // CPU-to-GPU (Grace Hopper / Blackwell Grace)
    { id: 'c2c', label: 'NVLink-C2C (Grace Hopper)', bw: 900, scope: 'intra' },  // 900 GB/s bidirectional

    // PCIe (Host-to-GPU / GPU-to-NIC)
    { id: 'pcie6', label: 'PCIe 6.0', bw: 256, scope: 'intra' },                  // 256 GB/s bidirectional (x16)
    { id: 'pcie5', label: 'PCIe 5.0', bw: 128, scope: 'intra' },                  // 128 GB/s bidirectional (x16)
    { id: 'pcie4', label: 'PCIe 4.0', bw: 64, scope: 'intra' },                   // 64 GB/s bidirectional (x16)
]

// Inter-node Connections （跨节点)
export const INTER_NODES_CONNECTION = [
    // NVLink Network (Rack-scale / Cross-node NVLink expansion, e.g., GB200 NVL72)
    { id: 'nvlink_switch_net', label: 'NVLink Network (NVL72 Cluster)', bw: 1800, scope: 'inter' },

    // InfiniBand (NVIDIA Quantum)
    { id: 'ib_x800', label: 'InfiniBand X800 (Quantum-X800)', bw: 100, scope: 'inter' }, // 800 Gbps (~100 GB/s unidirectional)
    { id: 'ib_ndr', label: 'InfiniBand NDR (Quantum-2)', bw: 50, scope: 'inter' },        // 400 Gbps (~50 GB/s unidirectional)
    { id: 'ib_hdr', label: 'InfiniBand HDR (Quantum-1)', bw: 25, scope: 'inter' },        // 200 Gbps (~25 GB/s unidirectional)
    { id: 'ib_edr', label: 'InfiniBand EDR', bw: 12.5, scope: 'inter' },                  // 100 Gbps (~12.5 GB/s unidirectional)

    // Ethernet / RoCE (NVIDIA Spectrum)
    { id: 'eth_sn5000', label: 'Spectrum-X Spectrum-4 (800G Ethernet)', bw: 100, scope: 'inter' }, // 800 Gbps
    { id: 'eth_sn4000', label: 'Spectrum-3 (400G Ethernet)', bw: 50, scope: 'inter' },             // 400 Gbps
]