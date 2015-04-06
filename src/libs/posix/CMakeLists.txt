set(gmac_os_SRC 
    io.cpp
    posix.h
    posix.cpp
)

add_gmac_sources(gmac-hpe-api ${gmac_os_SRC})
