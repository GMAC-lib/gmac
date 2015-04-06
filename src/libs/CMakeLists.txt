add_subdirectory(mpi)

set(gmac_lib_SRC
    common.cpp
    posix/io.cpp
    posix/posix.h
    posix/posix.cpp
    stdc/io.cpp
    stdc/stdc.h
    stdc/stdc.cpp
)

add_gmac_sources(gmac-libs ${gmac_lib_SRC})
