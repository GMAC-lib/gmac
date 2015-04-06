set(common_ROOT ${CMAKE_SOURCE_DIR}/tests/common)
include_directories(${common_ROOT})

set(launcher_SRC Test.h Test.cpp Variable.h common.h file.cpp main.cpp ${OS_DIR}/Test.cpp)
add_executable(launcher ${launcher_SRC})
target_link_libraries(launcher test-common ${CMAKE_THREAD_LIBS_INIT})
