set(core_hpe_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/Accelerator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Accelerator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AddressSpace.h
    ${CMAKE_CURRENT_SOURCE_DIR}/AddressSpace.cpp  
    ${CMAKE_CURRENT_SOURCE_DIR}/Mode.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Mode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Context.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Process.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Process.cpp    
    ${CMAKE_CURRENT_SOURCE_DIR}/Queue.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Queue.cpp  
    ${CMAKE_CURRENT_SOURCE_DIR}/Thread.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Thread.cpp
)  


# Export tests one level up
set(unit_SRC ${unit_SRC}
    ${core_hpe_SRC}
    PARENT_SCOPE)
