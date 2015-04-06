add_subdirectory(dbc)
add_subdirectory(${API_DIR})

# Generate configuraion file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)

set(gmac_config_SRC
    common.h
    config.h
    dbc.h
    order.h
)

add_gmac_sources(gmac-config ${gmac_config_SRC})
add_gmac_sources(gmac-dbc)
