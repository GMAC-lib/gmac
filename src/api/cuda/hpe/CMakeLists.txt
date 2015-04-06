set(arch_hpe_api_SRC
    compat.cpp
    driver.cpp
)

set(arch_hpe_SRC
    cuda.cpp
    Accelerator.h
    Accelerator-impl.hs
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
    Module.h
    Module-impl.h
    Module.cpp
    memory/Bitmap.cpp
)

set(arch_hpe_DBC
    dbc/Accelerator.h
    dbc/Accelerator.cpp
    dbc/Mode.h
    dbc/Mode.cpp
)

add_gmac_sources(gmac-arch-hpe ${arch_hpe_SRC} ${arch_hpe_DBC})
add_gmac_sources(gmac-arch-hpe-api ${arch_hpe_api_SRC})

add_gmac_sources(gmac-memory ${arch_memory_SRC})
