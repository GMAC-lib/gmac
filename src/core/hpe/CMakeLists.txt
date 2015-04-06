set(core_hpe_SRC
    Accelerator.h
    Accelerator-impl.h
    Accelerator.cpp
    AddressSpace.h
    AddressSpace.cpp
    Context.h
    Context.cpp
    Descriptor.h
    Descriptor-impl.h
    Kernel.h
    Kernel-impl.h
    Mode.h
    Mode-impl.h
    Mode.cpp
    Process.h
    Process-impl.h
    Process.cpp
    Thread.h
    Thread-impl.h
    Queue.h
    Queue.cpp
)

set(core_hpe_DBC
    dbc/Accelerator.h
    dbc/Accelerator.cpp
    dbc/Context.h
    dbc/Context.cpp
    dbc/Kernel.h
    dbc/Kernel.cpp
    dbc/Mode.h
    dbc/Mode.cpp
    dbc/Process.h
    dbc/Process.cpp
)

add_gmac_sources(gmac-core-hpe ${core_hpe_SRC} ${core_hpe_DBC})
