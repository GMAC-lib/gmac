set(cuda_test_SRC
    Accelerator.h Accelerator.cpp
    Mode.h Mode.cpp
)

add_gmac_sources(gmac-dbc ${cuda_test_SRC})
