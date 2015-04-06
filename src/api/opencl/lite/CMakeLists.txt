message(STATUS "Compiling GMAC/Lite for OpenCL")

set(arch_lite_api_SRC
    opencl.cpp
)

set(arch_lite_SRC
    Error.cpp
    Mode.h
    Mode-impl.h
    Mode.cpp
    ModeMap.h
    ModeMap-impl.h
    Process.h
    Process.cpp
)

set(arch_lite_DBC
)

add_gmac_sources(gmac-arch-lite ${arch_lite_SRC} ${arch_lite_DBC})
add_gmac_sources(gmac-arch-lite-api ${arch_lite_api_SRC})
