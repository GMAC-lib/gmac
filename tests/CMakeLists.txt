remove_definitions(-DGMAC_DLL_EXPORTS)

include_directories(${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR}/src)
add_subdirectory(common)
add_subdirectory(launcher)
add_subdirectory(simple)
add_subdirectory(unit)
