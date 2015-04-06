set(util_os_SRC
    loader.h
    loader.cpp
    Lock.h
    Lock-impl.h
    Lock.cpp
    Private.h
    Private-impl.h
    Semaphore.h
    Semaphore-impl.h
    Semaphore.cpp
    Thread-impl.h
)

set(util_os_DBC
    dbc/Lock.h
    dbc/Lock.cpp
)

add_gmac_sources(gmac-util ${util_os_SRC} ${util_os_DBC})
