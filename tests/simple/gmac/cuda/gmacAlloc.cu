#include <stdio.h>
#include <gmac/cuda.h>

#include "utils.h"

__global__ void kernelFill(unsigned *A, unsigned off, size_t size)
{
    unsigned localIdx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned idx = localIdx + off;

    if (idx >= size) return;
    A[localIdx] = idx;
}

int main(int argc, char *argv[])
{
    const unsigned totalSize = 8 * 1024 * 1024;
    gmactime_t s, t;

    for (unsigned currentSize = totalSize; currentSize > 32; currentSize /= 2) {
        assert(totalSize % currentSize == 0);
        fprintf(stderr,"Size: %u\n", currentSize);
        size_t nObjects = totalSize / currentSize;
        unsigned **objects = (unsigned **) malloc(nObjects * sizeof(int *));
        assert(objects != NULL);

        getTime(&s);
        for(size_t i = 0; i < nObjects; i++) {
            assert(gmacMalloc((void **)&objects[i], currentSize * sizeof(int)) == gmacSuccess);
        }
        getTime(&t);
        printTime(&s, &t, "Alloc: ", "\n");

        getTime(&s);
        unsigned off = 0;
        dim3 Db(currentSize > 256? 256: currentSize);
        dim3 Dg(currentSize / Db.x);
        if (currentSize > 256 && currentSize % 256 != 0) Dg.x++;

        for(size_t i = 0; i < nObjects; i++) {
            kernelFill<<<Dg, Db>>>(gmacPtr(objects[i]), off, totalSize);
            off += currentSize;
        }
        gmacThreadSynchronize();
        getTime(&t);
        printTime(&s, &t, "Run: ", "\n");

        getTime(&s);
        off = 0;
        for(size_t i = 0; i < nObjects; i++) {
            for(size_t j = 0; j < currentSize; j++) {
                size_t idx = off + j;
                assert(objects[i][j] == idx);
            }
            off += currentSize;
        }
        getTime(&t);
        printTime(&s, &t, "Check: ", "\n");

        getTime(&s);
        for(size_t i = 0; i < nObjects; i++) {
            gmacFree(objects[i]);
        }
        free(objects);
        getTime(&t);
        printTime(&s, &t, "Free: ", "\n");
    }

    return 0;
}
