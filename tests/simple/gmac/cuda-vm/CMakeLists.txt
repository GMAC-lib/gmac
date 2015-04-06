set(common_ROOT ${CMAKE_SOURCE_DIR}/tests/common)

include_directories(${PROJECT_SOURCE_DIR}/src/include ${PROJECT_SOURCE_DIR}/src ${common_ROOT})

include(FindCUDA)

set(common_SRC
     ${common_ROOT}/debug.h
     ${common_ROOT}/utils.h ${common_ROOT}/utils-impl.h ${common_ROOT}/utils.cpp 
     ${common_ROOT}/barrier.h
     ${common_ROOT}/cycle.h)
set(common_os_SRC
        ${common_ROOT}/${OS_DIR}/utils.cpp
        ${common_ROOT}/${OS_DIR}/barrier.h
        ${common_ROOT}/${OS_DIR}/barrier.c)
# Set source group for common files
string(REPLACE "${CMAKE_SOURCE_DIR}" "" common_GROUP ${common_ROOT})
string(REGEX REPLACE "^/" "" common_GROUP ${common_GROUP})
string(REPLACE "/" "\\\\" common_GROUP ${common_GROUP})
source_group(${common_GROUP} FILES ${common_SRC})
source_group("${common_GROUP}\\${OS_DIR}" FILES ${common_os_SRC})
set(common_SRC ${common_SRC} ${common_os_SRC})

# Create a source group for current directory
string(REPLACE "${CMAKE_SOURCE_DIR}" "" current_GROUP ${CMAKE_CURRENT_SOURCE_DIR})
string(REGEX REPLACE "^/" "" current_GROUP ${current_GROUP})
string(REPLACE "/" "\\\\" current_GROUP ${current_GROUP})
source_group(${current_GROUP} FILES
    vmVecAdd.cu
    vmNull.cu
    vmStencil.cu
    vmMatrixMul.cu
)

message(STATUS "Size of pointer: ${CMAKE_SIZEOF_VOID_P}")
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(CUDA_64_BIT_DEVICE_CODE ON)
else()
    set(CUDA_64_BIT_DEVICE_CODE OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCC)
    set(CUDA_NVCC_FLAGS "--ptxas-options=-v" "-arch=sm_11" "--compiler-options;-fno-strict-aliasing")
endif(CMAKE_COMPILER_IS_GNUCC)

# vmVecAdd
cuda_compile(vecAdd vmVecAdd.cu)
add_executable(vmVecAdd ${vecAdd} ${common_SRC} vmVecAdd.cu)
target_link_libraries(vmVecAdd gmac)

# vmNull
cuda_compile(null vmNull.cu)
add_executable(vmNull ${null} ${common_SRC} vmNull.cu)
target_link_libraries(vmNull gmac)

# vmStencil
cuda_compile(stencil vmStencil.cu)
add_executable(vmStencil ${stencil} ${common_SRC} vmStencil.cu)
target_link_libraries(vmStencil gmac)

# vmMatrixMul
cuda_compile(matrixMul vmMatrixMul.cu)
add_executable(vmMatrixMul ${matrixMul} ${common_SRC} vmMatrixMul.cu)
target_link_libraries(vmMatrixMul gmac)

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
add_custom_target(InputSets ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../input
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/inputset)
endif(CMAKE_SYSTEM_NAME MATCHES "Windows")
