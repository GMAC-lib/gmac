add_subdirectory(amd)
add_subdirectory(nvidia)

group_gmac_sources(gmac-arch-hpe-gpu gmac-arch-hpe-gpu-amd gmac-arch-hpe-gpu-nvidia)
