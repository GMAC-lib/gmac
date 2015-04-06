message(STATUS "Compiling helpers lib for OpenCL")

set(arch_libs_api_SRC
    helpers.cpp
)

set(arch_lite_DBC
)

add_gmac_sources(gmac-libs-lite-api ${arch_libs_api_SRC})
