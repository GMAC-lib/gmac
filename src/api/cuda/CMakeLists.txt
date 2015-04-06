add_subdirectory(hpe)

set(arch_SRC
    IOBuffer.h
    IOBuffer-impl.h
    IOBuffer.cpp
    Mode.h
    Tracer.h
    Tracer-impl.h
)

add_gmac_sources(gmac-arch ${arch_SRC})
add_gmac_sources(gmac-arch-hpe)
add_gmac_sources(gmac-arch-hpe-api)
