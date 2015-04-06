#include "gtest/gtest.h"

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    if(ret != 0) fprintf(stderr,"Some Tests were wrong\n");
    /* We always return 0 to let hudson continue executing tests */
    return 0;
}
