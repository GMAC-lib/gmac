#include <gmac/cuda.h>

#include "utils.h"
#include "debug.h"

const char *vecSizeStr = "GMAC_VECSIZE";
const size_t vecSizeDefault = 1 * 1024 * 1024;

size_t vecSize = 0;
const size_t blockSize = 512;

const char *msg = "Done!";

__global__ void vecInc(float *a, size_t size)
{
    int i =  threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= size) return;

    a[i] += 1.f;
}

#define ITER 10

int main(int argc, char *argv[])
{
    float *a = NULL;
    gmactime_t s, t;

    setParam<size_t>(&vecSize, vecSizeStr, vecSizeDefault);

    gmacMigrate(0);
    getTime(&s);
    // Alloc & init input data
    if(gmacMalloc((void **)&a, vecSize * sizeof(float)) != gmacSuccess)
        CUFATAL();
    gmacMemset(a, 0, vecSize * sizeof(float));
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");
    
    // Call the kernel
    getTime(&s);
    dim3 Db(blockSize);
    dim3 Dg((unsigned long)vecSize / blockSize);
    if(vecSize % blockSize) Dg.x++;

    for(int i = 0; i < ITER; i++) {
        vecInc<<<Dg, Db>>>(gmacPtr(a), vecSize);
        if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
    }
    getTime(&t);
    printTime(&s, &t, "Run: ", "\n");

    gmacMemset(a, 0, vecSize * sizeof(float));
    getTime(&s);
    for(unsigned i = 0; i < ITER; i++) {
        vecInc<<<Dg, Db>>>(gmacPtr(a), vecSize);
        if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
        gmacMigrate(i % 2);
    }
    getTime(&t);
    printTime(&s, &t, "Run: ", "\n");

    float error = 0.f;
    for(unsigned i = 0; i < vecSize; i++) {
        error += a[i] - float(ITER);
    }
    getTime(&t);
    printTime(&s, &t, "Check: ", "\n");

    getTime(&s);
    gmacFree(a);
    getTime(&t);
    printTime(&s, &t, "Free: ", "\n");

    return error != 0.f;
}
