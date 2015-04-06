set(arch_hpe_gpu_nvidia_SRC
    Accelerator.h
    Accelerator.cpp
)

set(arch_hpe_gpu_nvidia_DBC
)

add_gmac_sources(gmac-arch-hpe-gpu-nvidia ${arch_hpe_gpu_nvidia_SRC} ${arch_hpe_gpu_nvidia_DBC})
