
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
else(CMAKE_SYSTEM_NAME MATCHES "Windows")
    include(ExternalProject)
    ExternalProject_Add(
        openclcc
        URL http://openclcc.googlecode.com/files/openclcc-0.0.1-3.tar.gz
        SOURCE_DIR openclcc-0.0.1
        CONFIGURE_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/openclcc-0.0.1/configure --prefix=${CMAKE_CURRENT_BINARY_DIR}/openclcc-build --with-include=${OPENCL_INCLUDE}
    )

    add_custom_target(openclcc_install ALL
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/openclcc-build/bin/openclcc
        ${CMAKE_CURRENT_BINARY_DIR}/../)

    add_custom_target(gmacl-opencl-compile_install ALL
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/gmac-compile-cl
        ${CMAKE_CURRENT_BINARY_DIR}/../)
endif(CMAKE_SYSTEM_NAME MATCHES "Windows")
