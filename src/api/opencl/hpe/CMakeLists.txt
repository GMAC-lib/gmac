add_subdirectory(gpu)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gmac.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/gmac.cpp)

set(arch_hpe_api_SRC
    gmac.cpp
    compat.cpp
)

set(arch_hpe_SRC
    opencl.cpp
    Accelerator.h
    Accelerator-impl.h
    Accelerator.cpp
    Context.h
    Context-impl.h
    Context.cpp
    ContextFactory.h
    ContextFactory.cpp
    Error.cpp
    Kernel.h
    Kernel-impl.h
    Kernel.cpp
    Mode.h
    Mode-impl.h
    Mode.cpp
    ModeFactory.h
    ModeFactory.cpp
    cpu/Accelerator.h
    cpu/Accelerator.cpp
)

set(arch_hpe_DBC
    dbc/Accelerator.h   
    dbc/Accelerator.cpp
    dbc/Mode.h
    dbc/Mode.cpp
)

add_gmac_sources(gmac-arch-hpe-common ${arch_hpe_SRC} ${arch_hpe_DBC})
add_gmac_sources(gmac-arch-hpe-api ${arch_hpe_api_SRC})
add_gmac_sources(gmac-arch-hpe-gpu)
