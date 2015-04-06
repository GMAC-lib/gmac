if(DEFINED THREAD_DIR)
    add_subdirectory(${THREAD_DIR})
endif(DEFINED THREAD_DIR)

set(hpe_lib_SRC
    gmac.cpp
    init.h
    init.cpp
)

add_gmac_sources(gmac-hpe-api ${hpe_lib_SRC})
