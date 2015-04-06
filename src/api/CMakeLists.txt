add_subdirectory(${API_DIR})

add_gmac_sources(gmac-arch)
add_gmac_sources(gmac-arch-hpe)
add_gmac_sources(gmac-arch-hpe-api)

if(MAKE_CL)
    add_gmac_sources(gmac-arch-lite)
    add_gmac_sources(gmac-arch-lite-api)
    add_gmac_sources(gmac-libs-lite-api)
endif(MAKE_CL)
