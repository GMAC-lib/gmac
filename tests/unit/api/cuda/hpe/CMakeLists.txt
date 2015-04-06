set(opencl_SRC )

# Export tests one level up
set(unit_SRC ${unit_SRC}
    ${opencl_SRC}
    PARENT_SCOPE)
