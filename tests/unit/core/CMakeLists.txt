add_subdirectory(allocator)
add_subdirectory(hpe)
set(core_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/AllocationMap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/IOBuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/IOBuffer.cpp
)  


# Export tests one level up
set(unit_SRC ${unit_SRC}
    ${core_SRC}
    PARENT_SCOPE)
