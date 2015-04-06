set(util_os_SRC
    FileLock.h
    FileLock-impl.h
    FileLock.cpp
    loader.h
    Lock.h
    Lock-impl.h
    Lock.cpp
    Thread-impl.h
    Semaphore.h
    Semaphore-impl.h
    Semaphore.cpp
)

set(util_os_DBC
    dbc/Lock.h
    dbc/Lock.cpp
)

add_gmac_sources(gmac-util ${util_os_SRC} ${util_os_DBC})
