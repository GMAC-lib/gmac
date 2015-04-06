#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac/cuda.h>
#include <gmac/vm.h>

#include "debug.h"
#include "utils.h"

#define SIZE 1

const char *vecSizeStr = "GMAC_VECSIZE";
const size_t vecSizeDefault = 16 * 1024 * 1024;

size_t vecSize = 0;
const size_t blockSize = 512;

const char *msg = "Done!";

__global__ void vecAdd(float *c, float *a, float *b, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= size) return;
	__globalSt<float>(c + i, a[i] + b[i]);
}

int main(int argc, char *argv[])
{
	float *a, *b, *c;
	gmactime_t s, t;

	setParam<size_t>(&vecSize, vecSizeStr, vecSizeDefault);
	fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

    getTime(&s);
    // Alloc & init input data
    if(gmacMalloc((void **)&a, vecSize * sizeof(float)) != gmacSuccess)
        CUFATAL();
    if(gmacMalloc((void **)&b, vecSize * sizeof(float)) != gmacSuccess)
        CUFATAL();
    // Alloc output data
    if(gmacMalloc((void **)&c, vecSize * sizeof(float)) != gmacSuccess)
        CUFATAL();
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    getTime(&s);
    randInit(a, vecSize);
    randInit(b, vecSize);
    getTime(&t);
    printTime(&s, &t, "Init: ", "\n");
    
    // Call the kernel
    getTime(&s);
    dim3 Db(blockSize);
    dim3 Dg(unsigned(vecSize / blockSize));
    if(vecSize % blockSize) Dg.x++;
    vecAdd<<<Dg, Db>>>(gmacPtr(c), gmacPtr(a), gmacPtr(b), vecSize);
    if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
    getTime(&t);
    printTime(&s, &t, "Run: ", "\n");

    getTime(&s);
    float error = 0;
    for(unsigned i = 0; i < vecSize; i++) {
        error += c[i] - (a[i] + b[i]);
    }
    getTime(&t);
    printTime(&s, &t, "Check: ", "\n");

    fprintf(stderr,"Error: %f\n", error);

    gmacFree(a);
    gmacFree(b);
    gmacFree(c);

    return error != 0;
}
