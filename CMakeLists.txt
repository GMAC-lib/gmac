project(gmac)
cmake_minimum_required(VERSION 2.8)

include(gmac.cmake)
include(CheckIncludeFile)
include(CheckFunctionExists)
include(FindThreads)

set(GMAC_VERSION "11.12")

# Set the configuration being built
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Setup GMAC as main target
include_directories(${CMAKE_SOURCE_DIR}/src
                    ${CMAKE_BINARY_DIR}/src
                    ${CMAKE_SOURCE_DIR}/src/include
                    ${CMAKE_BINARY_DIR}/src/include)
# GMAC is a DLL
add_definitions(-DGMAC_DLL)
# Compiling GMAC
add_definitions(-DGMAC_DLL_EXPORTS)

# Identify thread library being used
if(CMAKE_USE_PTHREADS_INIT)
    message(STATUS "Using POSIX threads")
    add_definitions(-DHAVE_PTHREADS)
    set(gmac_LIBS ${gmac_LIBS} ${CMAKE_THREAD_LIBS_INIT})
    set(thread_LIB ${CMAKE_THREAD_LIBS_INIT})
    add_gmac_test_library(${CMAKE_THREAD_LIBS_INIT})
    set(THREAD_DIR "pthreads")
elseif(CMAKE_USE_WIN32_THREADS_INIT)
    message(STATUS "Using Windows threads")
else(CMAKE_USE_PTHREADS_INIT)
    message(FATAL "Your thread package (if any) is not supported")
endif(CMAKE_USE_PTHREADS_INIT)

# Identify operating system being used
option(USE_32BITS "Compile for 32 bits processors (Linux and Mac OS only)" OFF)
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    message(STATUS "Compiling for GNU/Linux")
    set(OS_DIR "posix")
    set(gmac_static_FLAGS "-fPIC")
    add_definitions(-DLINUX)
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
    if(USE_32BITS)
        message(STATUS "Compiling for 32 bits")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m32")
        set(CMAKE_LINKER_FLAGS "-m32")
    endif(USE_32BITS)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    message(STATUS "Compiling for Mac OS X")
    set(OS_DIR "posix")
    set(gmac_static_FLAGS "-fPIC")
    add_definitions(-DDARWIN)
    if(USE_32BITS)
        message(STATUS "Compiling for 32 bits")
        set (CMAKE_OSX_ARCHITECTURES i386)
    endif(USE_32BITS)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

# Add version-specific headers in the included files
set(GMAC_HEADERS ${CMAKE_BINARY_DIR}/src/include/gmac/api.h
                 ${CMAKE_BINARY_DIR}/src/include/gmac/shared_ptr
                 ${CMAKE_BINARY_DIR}/src/include/gmac/static
                 ${CMAKE_SOURCE_DIR}/src/include/gmac/new
                 ${CMAKE_SOURCE_DIR}/src/include/gmac/types.h
                 ${CMAKE_SOURCE_DIR}/src/include/gmac/visibility.h)
set(GMAC_TARGET_NAME "gmac")

include("CMakeLists-windows.txt" OPTIONAL)
include("CMakeLists-opencl.txt" OPTIONAL)
include("CMakeLists-cuda.txt")

# Check for Tracing
option(USE_TRACE_CONSOLE "Enable producing traces in the console" OFF)
option(USE_TRACE_PARAVER "Enable producing Paraver traces" OFF)
option(USE_TRACE_LOCKS "Enable producing traces for locks" OFF)
if(USE_TRACE_CONSOLE)
    if(USE_TRACE_LOCKS)
        add_definitions(-DUSE_TRACE_CONSOLE -DUSE_TRACE -DUSE_TRACE_LOCKS)
    else(USE_TRACE_LOCKS)
        add_definitions(-DUSE_TRACE_CONSOLE -DUSE_TRACE)
    endif(USE_TRACE_LOCKS)
elseif(USE_TRACE_PARAVER)
    if(USE_TRACE_LOCKS)
        add_definitions(-DUSE_TRACE_PARAVER -DUSE_TRACE -DUSE_TRACE_LOCKS)
    else(USE_TRACE_LOCKS)
        add_definitions(-DUSE_TRACE_PARAVER -DUSE_TRACE)
    endif(USE_TRACE_LOCKS)
endif(USE_TRACE_CONSOLE)

# Setup configuration options
option(USE_MULTI_CONTEXT "Enable multi context" OFF)
if(USE_MULTI_CONTEXT)
    message(STATUS "Using multi context")
    add_definitions(-DUSE_MULTI_CONTEXT)
endif(USE_MULTI_CONTEXT)

# Setup configuration options
option(USE_MPI "Enable MPI interposition" OFF)
if(USE_MPI)
    message(STATUS "Using MPI")
    add_definitions(-DUSE_MPI)
endif(USE_MPI)


# Setup configuration options
option(USE_VM "Enable software GPU virtual memory" OFF)
if(USE_VM)
    message(STATUS "Compiling with VM support")
    add_definitions(-DUSE_VM)
    set(USE_VM_BITMAP "1")
    if(VM_BITMAP MATCHES "bit")
        message(STATUS "Using 'bit' bitmap implementation")
        add_definitions(-DBITMAP_BIT)
    else(VM_BITMAP MATCHES "bit")
        message(STATUS "Using 'byte' bitmap implementation")
        add_definitions(-DBITMAP_BYTE)
    endif(VM_BITMAP MATCHES "bit")

    if(USE_HOSTMAP_VM)
        add_definitions(-DUSE_HOSTMAP_VM)
    endif(USE_HOSTMAP_VM)
else(USE_VM)
    set(USE_VM_BITMAP "0")
endif(USE_VM)

# Setup configuration options
option(USE_INTERNAL_API "Enable internal functions to tweak GMAC" OFF)
if(USE_INTERNAL_API)
    message(STATUS "Compiling with internal API support")
    add_definitions(-DUSE_INTERNAL_API)
    set(USE_INTERNAL_API_HEADER "1")
else(USE_INTERNAL_API)
    set(USE_INTERNAL_API_HEADER "0")
endif(USE_INTERNAL_API)

# Enable features that depend on std::tr1
option(USE_TR1 "Enable features that depend on std::tr1" ON)
if(USE_TR1)
    message(STATUS "Compiling with internal API support")
    add_definitions(-DUSE_TR1)
    set(USE_TR1_HEADER "1")
else(USE_TR1)
    set(USE_TR1_HEADER "0")
endif(USE_TR1)

# Enable features that depend on boost
option(USE_BOOST "Enable features that depend on std::tr1" OFF)
if(USE_BOOST)
    message(STATUS "Compiling with internal API support")
    add_definitions(-DUSE_BOOST)
    set(USE_BOOST_HEADER "1")
else(USE_BOOST)
    set(USE_BOOST_HEADER "0")
endif(USE_BOOST)

option(USE_SUBBLOCK_TRACKING "Enable subblock modification tracking" OFF)
if(USE_SUBBLOCK_TRACKING)
    message(STATUS "Compiling with subblock modification tracking")
    add_definitions(-DUSE_SUBBLOCK_TRACKING)
endif(USE_SUBBLOCK_TRACKING)


# Make sure we compile the necessary libraries
set(CMAKE_CXX_FLAGS ${CMAKE_C_FLAGS})
add_subdirectory(lib)
import_gmac_libraries()

# Configure compiler flags for different configurations
if(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3 -g -fomit-frame-pointer")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -g -DDEBUG -O0")
    option(USE_COVERAGE "Enable code coverage information gathering" OFF)
    if(USE_COVERAGE)
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage")
    endif(USE_COVERAGE)
elseif(MSVC)
    set(CMAKE_C_FLAGS " /W3 /WX /EHsc")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} /Zi /O2 /MD")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} /DDEBUG /Zi /Od /MDd /Gs")
endif(CMAKE_COMPILER_IS_GNUCC)

if(CMAKE_COMPILER_IS_GNUCC)
    option(USE_CXX0X "Enable experimental support provided by the C++0x standard" OFF)
    if(USE_CXX0X)
        set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++0x -DUSE_CXX0X")
    endif(USE_CXX0X)
endif(CMAKE_COMPILER_IS_GNUCC)

set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})

# Check for header files
check_function_exists(llabs HAVE_LLABS)
check_function_exists(getline HAVE_GETLINE)
if(HAVE_GETLINE)
    add_definitions(-DHAVE_GETLINE)
endif(HAVE_GETLINE)

# Setup extra libraries needed by GMAC
if(${OS_DIR} MATCHES "posix")
    message(STATUS "Compiling for POSIX")
    add_definitions(-DPOSIX)
    set(dl_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}dl${CMAKE_SHARED_LIBRARY_SUFFIX}")
    find_library(dl_LIB ${dl_NAME})
    if(NOT dl_LIB)
        message(FATAL "${dl_NAME} not found")
    endif(NOT dl_LIB)

    set(gmac_LIBS ${gmac_LIBS} ${dl_LIB})
    check_function_exists(posix_memalign HAVE_POSIX_MEMALIGN)
elseif(${OS_DIR} MATCHES "windows")
endif(${OS_DIR} MATCHES "posix")

# Add subdirectories
add_subdirectory(src)
add_gmac_groups(${gmac_SRC})


# Check if doc must be generated
option(MAKE_DOC "Generate GMAC documentation" OFF)
if(MAKE_DOC)
    message(STATUS "Generate GMAC documentation")
    add_subdirectory(doc)
endif(MAKE_DOC)

link_directories(${gmac_LIBDIR})

option(MAKE_STATIC "Build GMAC as a static library")
include("CMakeLists-libs.txt")
if(MAKE_STATIC)
    add_gmac_library(${GMAC_TARGET_NAME} STATIC ${gmac-hpe_SRC} ${gmac-include_SRC})
else(MAKE_STATIC)
    add_gmac_library(${GMAC_TARGET_NAME} SHARED ${gmac-hpe_SRC} ${gmac-include_SRC})
endif(MAKE_STATIC)
target_link_libraries(${GMAC_TARGET_NAME} gmac-common gmac-trace gmac-core gmac-core-hpe gmac-memory gmac-core-hpe gmac-libs ${gmac_LIBS})

set_target_properties(${GMAC_TARGET_NAME} PROPERTIES
    VERSION ${GMAC_VERSION} SOVERSION 1
)

include("CMakeLists-cl.txt" OPTIONAL)
# set_property(TARGET ${GMAC_TARGET_NAME} PROPERTY PUBLIC_HEADER ${GMAC_HEADERS})
message(STATUS "Compiling: ${GMAC_TARGET_NAME}")


if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
install(TARGETS ${GMAC_TARGET_NAME}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        COMPONENT libraries)

install(TARGETS ${GMAC_TARGET_NAME}
        RUNTIME DESTINATION bin
        COMPONENT runtime)

else(${CMAKE_SYSTEM_NAME} MATCHES "Windows")

install(TARGETS ${GMAC_TARGET_NAME}
        LIBRARY DESTINATION lib
        COMPONENT libraries)

endif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")

install(FILES ${GMAC_HEADERS}
        DESTINATION include/gmac
        COMPONENT headers)


# Check if tests must be compiled
option(MAKE_TESTS "Compile GMAC tests" OFF)
if(MAKE_TESTS)
    message(STATUS "Compile GMAC tests")
    add_subdirectory(tests)
    # Add libraries used in testing
endif(MAKE_TESTS)

# Check if tools must be compiled
option(MAKE_TOOLS "Compile GMAC tools" OFF)
if(MAKE_TOOLS)
    message(STATUS "Compile GMAC tools")
    add_subdirectory(tools)
endif(MAKE_TOOLS)

include("CMakeLists-opencl-install.txt" OPTIONAL)
include("CMakeLists-install.txt")
