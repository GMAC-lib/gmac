add_subdirectory(hpe)

set(core_SRC
    IOBuffer.h
    IOBuffer-impl.h
    Mode.h
    Mode-impl.h
    Process.h
    Process.cpp
    Thread.h
    Thread-impl.h
    Thread.cpp
)

set(core_DBC
    dbc/AllocationMap.h
    dbc/AllocationMap.cpp
    dbc/IOBuffer.h
    dbc/IOBuffer.cpp
)

add_gmac_sources(gmac-core ${core_SRC} ${core_DBC})
add_gmac_sources(gmac-core-hpe)
