set(memory_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/Manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Object.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Slab.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/ObjectMap.cpp)

# Export tests one level up
set(unit_SRC ${unit_SRC}
    ${memory_SRC}
    PARENT_SCOPE)
