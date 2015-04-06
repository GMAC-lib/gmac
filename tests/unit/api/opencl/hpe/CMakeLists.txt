set(opencl_SRC 
    ${CMAKE_CURRENT_SOURCE_DIR}/Context.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Context.cpp
)

# Export tests one level up
set(unit_SRC ${unit_SRC}
    ${opencl_SRC}
    PARENT_SCOPE)
