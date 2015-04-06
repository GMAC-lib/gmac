function(add_gmac_groups)
    foreach(__file ${ARGV})
        # Create group name
        string(REGEX REPLACE "/[^/]+$" "" file_DIR ${__file})
        if(${file_DIR} MATCHES ${CMAKE_SOURCE_DIR})
            string(REPLACE "${CMAKE_SOURCE_DIR}" "" group_DIR ${file_DIR})
        elseif(${file_DIR} MATCHES ${CMAKE_BINARY_DIR})
            string(REPLACE "${CMAKE_BINARY_DIR}" "" group_DIR ${file_DIR})
        endif(${file_DIR} MATCHES ${CMAKE_SOURCE_DIR})
        string(REPLACE "/" "\\\\" group_LABEL ${group_DIR})
        source_group(${group_LABEL} FILES ${__file})
    endforeach()
endfunction(add_gmac_groups)

macro(add_gmac_sources label)
    set(__sources "")
    foreach(__file ${ARGN})
        set(__source __source-NOTFOUND)
        find_file(__source ${__file} ${CMAKE_CURRENT_SOURCE_DIR} NO_DEFAULT_PATH)
        if(EXISTS ${__source})
            set(__sources ${__sources} ${__source})
        else()
            set(__binary __binary-NOTFOUND)
            find_file(__binary ${__file} ${CMAKE_CURRENT_BINARY_DIR} NO_DEFAULT_PATH)
            if(EXISTS ${__binary})
                set(__sources ${__sources} ${__binary})
            endif()
        endif()
    endforeach()

    # Add the source files to the global list
    set(${label}_SRC ${${label}_SRC} ${__sources} PARENT_SCOPE)
endmacro(add_gmac_sources)

macro(add_gmac_library name type)
    add_library(${name} ${type} ${ARGN})
    add_gmac_groups(${ARGN})
    set_target_properties(${name} PROPERTIES
        COMPILE_FLAGS ${gmac_static_FLAGS}
	COMPILE_DEFINITIONS_DEBUG USE_DBC
    )
endmacro(add_gmac_library)

macro(group_gmac_sources label)
    set(__sources "")
    foreach(__label ${ARGN})
        set(__sources ${__sources} ${${__label}_SRC})
    endforeach()

    # Add the source files to the global list
    set(${label}_SRC ${__sources} PARENT_SCOPE)
endmacro(group_gmac_sources)

function(add_gmac_test_include)
    get_property(gmac_test_INCLUDE GLOBAL PROPERTY gmac_test_INCLUDE)
    set(gmac_test_INCLUDE ${gmac_test_INCLUDE} ${ARGV})
    set_property(GLOBAL PROPERTY gmac_test_INCLUDE ${gmac_test_INCLUDE})
endfunction(add_gmac_test_include)

function(add_gmac_test_library)
    get_property(gmac_test_LIB GLOBAL PROPERTY gmac_test_LIB)
    set(gmac_test_LIB ${gmac_test_LIB} ${ARGV})
    set_property(GLOBAL PROPERTY gmac_test_LIB ${gmac_test_LIB})
endfunction(add_gmac_test_library)

macro(import_gmac_libraries)
    get_property(gmac_test_INCLUDE GLOBAL PROPERTY gmac_test_INCLUDE)
    get_property(gmac_test_LIB GLOBAL PROPERTY gmac_test_LIB)
endmacro(import_gmac_libraries)

