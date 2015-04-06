add_subdirectory(hpe)
add_subdirectory(helpers)
add_subdirectory(lite)

set(arch_SRC
    IOBuffer.h
    IOBuffer-impl.h
    Mode.h
    Tracer.h
    Tracer-impl.h
    opencl_utils.h
    opencl_utils.cpp
    opencl_utils-impl.h
)

add_gmac_sources(gmac-arch ${arch_SRC})

group_gmac_sources(gmac-arch-hpe gmac-arch-hpe-common gmac-arch-hpe-gpu)
add_gmac_sources(gmac-arch-hpe-api)

if(MAKE_CL)
    add_gmac_sources(gmac-arch-lite)
    add_gmac_sources(gmac-arch-lite-api)
    add_gmac_sources(gmac-libs-lite-api)
endif(MAKE_CL)
