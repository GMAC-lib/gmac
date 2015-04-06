add_subdirectory(${OS_DIR})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Parameter-def.h.in ${CMAKE_CURRENT_BINARY_DIR}/Parameter-def.h)

set(util_SRC
    Atomics.h
    FileLock.h
    GMACBase.h
    GMACBase.cpp
    Lock.h
    Lock-impl.h
    Lock.cpp
    Logger.h
    Logger-impl.h
    Logger.cpp
    NonCopyable.h
    Parameter.h
    Parameter-impl.h
    Parameter.cpp
    Parameter-def.h
    Private.h
    Reference.h
    Reference-impl.h
    ReusableObject.h
    Semaphore.h
    SharedPtr.h
    Singleton.h
    Singleton-impl.h
    Thread.h
    Unique.h
    Unique-impl.h
    UniquePtr.h
    allocator/Buddy.h
    allocator/Buddy.cpp
)

set(util_DBC
    allocator/dbc/Buddy.h
    allocator/dbc/Buddy.cpp
)

add_gmac_sources(gmac-util ${util_SRC} ${util_DBC})
