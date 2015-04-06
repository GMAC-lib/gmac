project(GMACSamples)
cmake_minimum_required(VERSION 2.8)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release")
endif(NOT CMAKE_BUILD_TYPE)

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    message(STATUS "Compiling for Windows")
    set(CMAKE_USE_RELATIVE_PATHS true)
    set(CMAKE_SUPPRESS_REGENERATION true)
	set(OS_DIR "windows")
    set(GMAC_HPE ../../lib/${CMAKE_IMPORT_LIBRARY_PREFIX}gmac-hpe)
    set(GMAC_CL ../../lib/${CMAKE_IMPORT_LIBRARY_PREFIX}gmac-cl)
else(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	set(OS_DIR "posix")
    find_library(GMAC_HPE gmac-hpe HINT /usr/loca/lib)
    find_library(GMAC_CL gmac-cl  HINT /usr/loca/lib)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    find_library(OPENCL OpenCL
         PATHS ENV "AMDAPPSDKROOT" "CUDA_LIB_PATH"
         PATH_SUFFIXES lib/x86_64)
else(CMAKE_SIZEOF_VOID_P EQUAL 8)
    find_library(OPENCL OpenCL
            PATHS ENV "AMDAPPSDKROOT" "CUDA_LIB_PATH"
        PATH_SUFFIXES /lib/x86)
endif(CMAKE_SIZEOF_VOID_P EQUAL 8)

include_directories(../include)
include_directories(common)

add_executable(eclVecAdd
	${CMAKE_SOURCE_DIR}/ecl/eclVecAdd.cpp
)
target_link_libraries(eclVecAdd ${GMAC_HPE})

add_executable(eclThreadVecAdd
	${CMAKE_SOURCE_DIR}/ecl/eclThreadVecAdd.cpp
	${CMAKE_SOURCE_DIR}/common/utils.h
	${CMAKE_SOURCE_DIR}/common/${OS_DIR}/utils.cpp
)
target_link_libraries(eclThreadVecAdd ${GMAC_HPE})

add_executable(eclBinarySearch
    ${CMAKE_SOURCE_DIR}/ecl/eclBinarySearch.cpp 
    ${CMAKE_SOURCE_DIR}/ecl/eclBinarySearchKernel.cl
)
target_link_libraries(eclBinarySearch ${GMAC_HPE})

add_executable(eclBinarySearch_cplusplus
    ${CMAKE_SOURCE_DIR}/ecl/eclBinarySearch_cplusplus.cpp 
    ${CMAKE_SOURCE_DIR}/ecl/eclBinarySearchKernel.cl
)
target_link_libraries(eclBinarySearch_cplusplus ${GMAC_HPE})

add_executable(eclBitonicSort
    ${CMAKE_SOURCE_DIR}/ecl/eclBitonicSort.cpp 
    ${CMAKE_SOURCE_DIR}/ecl/eclBitonicSortKernel.cl
)
target_link_libraries(eclBitonicSort ${GMAC_HPE})

add_executable(eclBitonicSort_cplusplus
    ${CMAKE_SOURCE_DIR}/ecl/eclBitonicSort_cplusplus.cpp 
    ${CMAKE_SOURCE_DIR}/ecl/eclBitonicSortKernel.cl
)
target_link_libraries(eclBitonicSort_cplusplus ${GMAC_HPE})

add_executable(eclBlackScholes
    ${CMAKE_SOURCE_DIR}/ecl/eclBlackScholes.cpp 
    ${CMAKE_SOURCE_DIR}/ecl/eclBlackScholesKernel.cl
)
target_link_libraries(eclBlackScholes ${GMAC_HPE})

add_executable(clVecAdd
    ${CMAKE_SOURCE_DIR}/cl/clVecAdd.cpp
)
target_link_libraries(clVecAdd ${GMAC_CL} ${OPENCL})

add_executable(clVecAdd_cplusplus
    ${CMAKE_SOURCE_DIR}/cl/clVecAdd_cplusplus.cpp
)
target_link_libraries(clVecAdd_cplusplus ${GMAC_CL} ${OPENCL})

add_executable(clBinarySearch
    ${CMAKE_SOURCE_DIR}/cl/clBinarySearch.cpp 
    ${CMAKE_SOURCE_DIR}/cl/clBinarySearchKernel.cl
)
target_link_libraries(clBinarySearch ${GMAC_CL} ${OPENCL})

add_executable(clBitonicSort
    ${CMAKE_SOURCE_DIR}/cl/clBitonicSort.cpp 
    ${CMAKE_SOURCE_DIR}/cl/clBitonicSortKernel.cl
)
target_link_libraries(clBitonicSort ${GMAC_CL} ${OPENCL})

add_executable(clBlackScholes
    ${CMAKE_SOURCE_DIR}/cl/clBlackScholes.cpp 
    ${CMAKE_SOURCE_DIR}/cl/clBlackScholesKernel.cl
)
target_link_libraries(clBlackScholes ${GMAC_CL} ${OPENCL})

