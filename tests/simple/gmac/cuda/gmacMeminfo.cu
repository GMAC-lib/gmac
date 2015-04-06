#include <stdio.h>
#include <gmac/cuda.h>

const unsigned allocations  = 10;
const size_t allocationSize = 4 * 1024 * 1024;

int main(int argc, char *argv[])
{
    void *dummy[allocations];
    size_t freeMem;
    gmacError_t ret;
    ret = gmacGetFreeMemory(gmacGetCurrentAcceleratorId(), &freeMem);
    assert(ret == gmacSuccess);
    fprintf(stdout, "Free memory: %zd\n", freeMem);

    for (unsigned i = 0; i < allocations; i++) {
        assert(gmacMalloc((void **)&dummy[i], allocationSize * sizeof(long)) == gmacSuccess);
        ret = gmacGetFreeMemory(gmacGetCurrentAcceleratorId(), &freeMem);
        assert(ret == gmacSuccess);
        fprintf(stdout, "Free memory: %zd\n", freeMem);
    }
    
    for (unsigned i = 0; i < allocations; i++) {
        assert(gmacFree(dummy[i]) == gmacSuccess);
        ret = gmacGetFreeMemory(gmacGetCurrentAcceleratorId(), &freeMem);
        assert(ret == gmacSuccess);
        fprintf(stdout, "Free memory: %zd\n", freeMem);
    }

    gmacMigrate(1);

    ret = gmacGetFreeMemory(gmacGetCurrentAcceleratorId(), &freeMem);
    assert(ret == gmacSuccess);
    fprintf(stdout, "Free memory: %zd\n", freeMem);

    for (unsigned i = 0; i < allocations; i++) {
        assert(gmacMalloc((void **)&dummy[i], allocationSize * sizeof(long)) == gmacSuccess);
        ret = gmacGetFreeMemory(gmacGetCurrentAcceleratorId(), &freeMem);
        assert(ret == gmacSuccess);
        fprintf(stdout, "Free memory: %zd\n", freeMem);
    }
    
    for (unsigned i = 0; i < allocations; i++) {
        assert(gmacFree(dummy[i]) == gmacSuccess);
        ret = gmacGetFreeMemory(gmacGetCurrentAcceleratorId(), &freeMem);
        assert(ret == gmacSuccess);
        fprintf(stdout, "Free memory: %zd\n", freeMem);
    }

    return 0;
}
