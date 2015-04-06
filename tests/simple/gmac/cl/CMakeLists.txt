set(common_ROOT ${CMAKE_SOURCE_DIR}/tests/common)

include_directories(${PROJECT_SOURCE_DIR}/src/include ${PROJECT_BINARY_DIR}/src/include ${common_ROOT})

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
    clVecAdd.cpp
    clVecAdd_cplusplus.cpp
    clStencil.cpp
    clStencilCommon.h
    clMatrixMul.cpp
    clMatrixMulKernel.cl
	clVecAdd_Helper.cpp
	clThreadVecAdd.cpp
	clBinarySearch.cpp	
	clBinarySearchKernel.cl
	clBinomialOption.cpp
	clBinomialOptionKernel.cl
	clBitonicSort.cpp
	clBitonicSortKernel.cl
	clBlackScholes.cpp
	clBlackScholesKernel.cl
	clThreadBinomialOption.cpp	
	clThreadBinomialOptionKernel.cl
	clMonteCarloAsian.cpp
	clMonteCarloAsianKernel.cl
	clThreadMonteCarloAsian.cpp
	clThreadMonteCarloAsianKernel.cl
)

add_executable(clVecAdd ${common_SRC} clVecAdd.cpp)
target_link_libraries(clVecAdd ${GMAC_CL})

add_executable(clThreadVecAdd ${common_SRC} clThreadVecAdd.cpp)
target_link_libraries(clThreadVecAdd ${GMAC_CL})

add_executable(clVecAdd_Helper ${common_SRC} clVecAdd_Helper.cpp)
target_link_libraries(clVecAdd_Helper ${GMAC_CL} ${GMAC_CL_HELPERS})

#add_executable(clVecAdd_cplusplus ${common_SRC} clVecAdd_cplusplus.cpp)
#target_link_libraries(clVecAdd_cplusplus ${GMAC_CL})

add_executable(clStencil ${common_SRC} clStencil.cpp clStencilCommon.h)
target_link_libraries(clStencil ${GMAC_CL})

add_executable(clMatrixMul ${common_SRC} clMatrixMul.cpp clMatrixMulKernel.cl)
target_link_libraries(clMatrixMul ${GMAC_CL})

add_executable(clBinarySearch ${common_SRC} clBinarySearch.cpp clBinarySearchKernel.cl)
target_link_libraries(clBinarySearch ${GMAC_CL})

add_executable(clBinomialOption ${common_SRC} clBinomialOption.cpp clBinomialOptionKernel.cl)
target_link_libraries(clBinomialOption ${GMAC_CL})

add_executable(clBitonicSort ${common_SRC} clBitonicSort.cpp clBitonicSortKernel.cl)
target_link_libraries(clBitonicSort ${GMAC_CL})

add_executable(clBlackScholes ${common_SRC} clBlackScholes.cpp clBlackScholesKernel.cl)
target_link_libraries(clBlackScholes ${GMAC_CL})

add_executable(clMonteCarloAsian ${common_SRC} clMonteCarloAsian.cpp clMonteCarloAsianKernel.cl)
target_link_libraries(clMonteCarloAsian ${GMAC_CL})

add_executable(clThreadBinomialOption ${common_SRC} clThreadBinomialOption.cpp clThreadBinomialOptionKernel.cl)
target_link_libraries(clThreadBinomialOption ${GMAC_CL})

add_executable(clThreadMonteCarloAsian ${common_SRC} clThreadMonteCarloAsian.cpp clThreadMonteCarloAsianKernel.cl)
target_link_libraries(clThreadMonteCarloAsian ${GMAC_CL})

file(COPY vars-cl.spec tests-cl.spec DESTINATION ${PROJECT_BINARY_DIR})

file(COPY clVecAddKernel.cl DESTINATION ${PROJECT_BINARY_DIR}/cl_kernels)

