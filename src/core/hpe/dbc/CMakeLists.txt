set(core_hpe_dbc_SRC
    Accelerator.h Accelerator.cpp
    Context.h Context.cpp
    Kernel.h Kernel.cpp
    Mode.h Mode.cpp
    Process.h Process.cpp)

add_gmac_sources(gmac-dbc ${core_hpe_dbc_SRC})
