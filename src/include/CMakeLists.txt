configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gmac/api.h.in ${CMAKE_CURRENT_BINARY_DIR}/gmac/api.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gmac/opencl_types.h.in ${CMAKE_CURRENT_BINARY_DIR}/gmac/opencl_types.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gmac/opencl.h.in ${CMAKE_CURRENT_BINARY_DIR}/gmac/opencl.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gmac/opencl.in ${CMAKE_CURRENT_BINARY_DIR}/gmac/opencl)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gmac/shared_ptr.in ${CMAKE_CURRENT_BINARY_DIR}/gmac/shared_ptr)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gmac/static.in ${CMAKE_CURRENT_BINARY_DIR}/gmac/static)

if(NOT USE_OPENCL)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gmac/vm.h.in ${CMAKE_CURRENT_BINARY_DIR}/gmac/vm.h)
endif(NOT USE_OPENCL)

set(gmac_include_SRC
    gmac/api.h.in
    gmac/cuda
    gmac/cuda.h
    gmac/cuda_types.h
    gmac/cl
    gmac/cl.h
    gmac/cl_types.h
    gmac/new
    gmac/opencl.in
    gmac/opencl.h.in
    gmac/opencl_types.h.in
    gmac/shared_ptr.in
    gmac/static.in
    gmac/types.h
    gmac/visibility.h
    gmac/vm.h.in
)

add_gmac_sources(gmac-include ${gmac_include_SRC})
