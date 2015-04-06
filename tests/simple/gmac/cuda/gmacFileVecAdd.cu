#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac/cuda.h>

#include "utils.h"
#include "debug.h"

#ifdef _MSC_VER
#define VECTORA "inputset\\vectorA"
#define VECTORB "inputset\\vectorB"
#define VECTORC "inputset\\vectorC"
#else
#define VECTORA "inputset/vectorA"
#define VECTORB "inputset/vectorB"
#define VECTORC "inputset/vectorC"
#endif

const size_t vecSize = 1024 * 1024;
const size_t blockSize = 512;

const char *msg = "Done!";

__global__ void vecAdd(float *c, float *a, float *b, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= size) return;

    c[i] = a[i] + b[i];
}


float doTest(float *a, float *b, float *c, float *orig, const char *name)
{
    gmactime_t s, t;
    static char buffer[1024];

    FILE * fA = fopen(VECTORA, "rb");
    assert(fA != NULL);
    FILE * fB = fopen(VECTORB, "rb");
    assert(fB != NULL);
    getTime(&s);
    size_t ret = fread(a, sizeof(float), vecSize, fA);
    assert(ret == vecSize);
    ret = fread(b, sizeof(float), vecSize, fB);
    assert(ret == vecSize);

    fclose(fA);
    fclose(fB);
    getTime(&t);
    snprintf(buffer, 1024, "%s-Init: ", name);
    printTime(&s, &t, buffer, "\n");

    // Call the kernel
    getTime(&s);
    dim3 Db(blockSize);
    dim3 Dg(vecSize / blockSize);
    if(vecSize % blockSize) Dg.x++;
    vecAdd<<<Dg, Db>>>(gmacPtr(c), gmacPtr(a), gmacPtr(b), vecSize);
    assert(gmacThreadSynchronize() == gmacSuccess);
    getTime(&t);
    snprintf(buffer, 1024, "%s-Run: ", name);
    printTime(&s, &t, buffer, "\n");

    getTime(&s);
    float error = 0.f;
    for(unsigned i = 0; i < vecSize; i++) {
        error += orig[i] - (c[i]);
    }
    getTime(&t);
    snprintf(buffer, 1024, "%s-Check: ", name);
    printTime(&s, &t, buffer, "\n");
    if(error != 0) fprintf(stderr, "Error: %f\n", error);

    return error;
}

int main(int argc, char *argv[])
{
    float *a, *b, *c;
    gmactime_t s, t;
    float error_shared, error_distributed, error_centralized;

    float * orig = (float *) malloc(vecSize * sizeof(float));
    FILE * fO = fopen(VECTORC, "rb");
    assert(fO != NULL);
    size_t ret = fread(orig, sizeof(float), vecSize, fO);
    assert(ret == vecSize);

    // Alloc output data
    assert(gmacMalloc((void **)&c, vecSize * sizeof(float)) == gmacSuccess);

    //////////////////////
    // Test shared objects
    //////////////////////
    getTime(&s);
    // Alloc & init input data
    assert(gmacMalloc((void **)&a, vecSize * sizeof(float)) == gmacSuccess);
    assert(gmacMalloc((void **)&b, vecSize * sizeof(float)) == gmacSuccess);
    getTime(&t);
    printTime(&s, &t, "Shared-Alloc: ", "\n");

    error_shared = doTest(a, b, c, orig, "Shared");

    getTime(&s);
    FILE * fC = fopen("vectorC_shared", "wb");
    assert(fC != NULL);
    ret = fwrite(c, sizeof(float), vecSize, fC);
    assert(ret == vecSize);
    fclose(fC);
    getTime(&t);
    printTime(&s, &t, "Shared-Write: ", "\n");

    getTime(&s);
    gmacFree(a);
    gmacFree(b);
    getTime(&t);
    printTime(&s, &t, "Shared-Free: ", "\n");

    //////////////////////////
    // Test replicated objects
    //////////////////////////
    getTime(&s);
    // Alloc & init input data
    assert(gmacGlobalMalloc((void **)&a, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_REPLICATED) == gmacSuccess);
    assert(gmacGlobalMalloc((void **)&b, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_REPLICATED) == gmacSuccess);
    getTime(&t);
    printTime(&s, &t, "Distributed-Alloc: ", "\n");

    error_distributed = doTest(a, b, c, orig, "Distributed");

    getTime(&s);
    fC = fopen("vectorC_replicated", "wb");
    ret = fwrite(c, sizeof(float), vecSize, fC);
    assert(ret == vecSize);
    fclose(fC);
    getTime(&t);
    printTime(&s, &t, "Distributed-Write: ", "\n");

    getTime(&s);
    gmacFree(a);
    gmacFree(b);
    getTime(&t);
    printTime(&s, &t, "Distributed-Free: ", "\n");

    ///////////////////////////
    // Test centralized objects
    ///////////////////////////
    getTime(&s);
    // Alloc & init input data
    assert(gmacGlobalMalloc((void **)&a, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_CENTRALIZED) == gmacSuccess);
    assert(gmacGlobalMalloc((void **)&b, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_CENTRALIZED) == gmacSuccess);
    getTime(&t);
    printTime(&s, &t, "Centralized-Alloc: ", "\n");

    error_centralized = doTest(a, b, c, orig, "Centralized");

    getTime(&s);
    fC = fopen("vectorC_centralized", "wb");
    ret = fwrite(c, sizeof(float), vecSize, fC);
    assert(ret == vecSize);
    fclose(fC);
    getTime(&t);
    printTime(&s, &t, "Centralized-Write: ", "\n");

    getTime(&s);
    gmacFree(a);
    gmacFree(b);
    getTime(&t);
    printTime(&s, &t, "Centralized-Write: ", "\n");

    gmacFree(c);
    free(orig);
    return error_shared != 0.f && error_distributed != 0.f && error_centralized != 0.f;
}
