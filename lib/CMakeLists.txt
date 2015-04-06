if(MAKE_TESTS)
    # Add Google Testing Framework Dependency
    set(gtest_DIR gtest-1.6.0)
    set(gtest_SRC
        ${gtest_DIR}/src/gtest-all.cc)

    include_directories(${gtest_DIR}/include ${gtest_DIR}
                        ${gmock_DIR}/include ${gmock_DIR})
    add_library(gtest STATIC ${gtest_SRC})
    add_gmac_test_include(${CMAKE_CURRENT_SOURCE_DIR}/${gtest_DIR}/include
                          ${CMAKE_CURRENT_SOURCE_DIR}/${gmock_DIR}/include) 
    add_gmac_test_library(gtest)
endif(MAKE_TESTS)
