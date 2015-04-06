set(paraver_SRC
    StreamOut.h
    Lock.h ${OS_DIR}/Lock.h ${OS_DIR}/Lock-impl.h
    Element.h Element-impl.h Element.cpp
    Names.h Names-impl.h Names.cpp
    Record.h Record-impl.h Record.cpp
    Trace.h Trace.cpp
    Pcf.h Pcf.cpp
)

add_gmac_sources(gmac-trace ${paraver_SRC})
