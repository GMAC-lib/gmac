set(core_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/Buddy.cpp)

# Export tests one level up
set(unit_SRC ${unit_SRC}
    ${core_SRC}
    PARENT_SCOPE)
