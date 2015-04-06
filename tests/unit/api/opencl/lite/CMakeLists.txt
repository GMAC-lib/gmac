# Export tests one level up
set(unit_SRC ${unit_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/ModeMap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/OpenCL.h
    ${CMAKE_CURRENT_SOURCE_DIR}/OpenCL.cpp
    PARENT_SCOPE)
