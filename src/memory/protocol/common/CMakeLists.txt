set(memory_protocol_common_SRC
    BlockList.h BlockList-impl.h
    BlockState.h BlockState-impl.h BlockState.cpp
    )

#set(memory_protocol_SRC
#    BlockList.h BlockList-impl.h
#    Lazy.h Lazy-impl.h Lazy.cpp
#    Gather.h Gather-impl.h Gather.cpp)

add_gmac_sources(gmac ${memory_protocol_common_SRC})
add_gmac_sources(gmac-dbc)
