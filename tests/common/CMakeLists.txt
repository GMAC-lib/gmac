if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    add_subdirectory(windows)
else(CMAKE_SYSTEM_NAME MATCHES "Windows")
    add_subdirectory(posix)
endif(CMAKE_SYSTEM_NAME MATCHES "Windows")

set(common_SRC ${common_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/cycle.h
    ${CMAKE_CURRENT_SOURCE_DIR}/barrier.h
    ${CMAKE_CURRENT_SOURCE_DIR}/debug.h
    ${CMAKE_CURRENT_SOURCE_DIR}/semaphore.h
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/utils-impl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp)

add_library(test-common STATIC ${common_SRC})
