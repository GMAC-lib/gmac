add_subdirectory(${OS_DIR})
if(USE_TRACE_PARAVER)
    add_subdirectory(paraver)
elseif(USE_TRACE_PPA)
    add_subdirectory(ppa)
endif(USE_TRACE_PARAVER)

set(trace_SRC
    Console.h
    Console.cpp
    Paraver.h
    Paraver.cpp
    Tracer.h
    Tracer-impl.h
    Tracer.cpp
)

add_gmac_sources(gmac-trace ${trace_SRC})
