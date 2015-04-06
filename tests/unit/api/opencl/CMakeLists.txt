add_subdirectory(hpe)
add_subdirectory(lite)

# Export tests one level up
set(unit_SRC ${unit_SRC}
    PARENT_SCOPE)
