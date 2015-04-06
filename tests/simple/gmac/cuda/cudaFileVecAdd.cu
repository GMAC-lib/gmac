#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>

#include <assert.h>

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
    float *c_dev;

    float *a_host = (float *)malloc(vecSize * sizeof(float));
    assert(a_host != NULL);
    float *b_host = (float *)malloc(vecSize * sizeof(float));
    assert(b_host != NULL);
    FILE * fA = fopen(VECTORA, "rb");
    assert(fA != NULL);
    FILE * fB = fopen(VECTORB, "rb");
    assert(fB != NULL);
    getTime(&s);
    size_t ret = fread(a_host, sizeof(float), vecSize, fA);
    assert(ret == vecSize);
    assert(cudaMemcpy(a, a_host, vecSize * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    free(a_host);
    fclose(fA);
    ret = fread(b_host, sizeof(float), vecSize, fB);
    assert(ret == vecSize);
    assert(cudaMemcpy(b, b_host, vecSize * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    free(b_host);
    fclose(fB);
    getTime(&t);
    assert(cudaMalloc((void **)&c_dev, vecSize * sizeof(float)) == cudaSuccess);
    snprintf(buffer, 1024, "%s:Init: ", name);
    printTime(&s, &t, buffer, "\n");

    // Call the kernel
    getTime(&s);
    dim3 Db(blockSize);
    dim3 Dg(vecSize / blockSize);
    if(vecSize % blockSize) Dg.x++;
    vecAdd<<<Dg, Db>>>(c_dev, a, b, vecSize);
    assert(cudaThreadSynchronize() == cudaSuccess);
    getTime(&t);
    snprintf(buffer, 1024, "%s:Run: ", name);
    printTime(&s, &t, buffer, "\n");

    getTime(&s);
    assert(cudaMemcpy(c, c_dev, vecSize * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    float error = 0.f;
    for(unsigned i = 0; i < vecSize; i++) {
        error += orig[i] - (c[i]);
    }
    cudaFree(c_dev);
    getTime(&t);
    snprintf(buffer, 1024, "%s:Check: ", name);
    printTime(&s, &t, buffer, "\n");

    return error;
}

int main(int argc, char *argv[])
{
    float *a, *b, *c;
    gmactime_t s, t;
    float error;

    float * orig = (float *) malloc(vecSize * sizeof(float));
    FILE * fO = fopen(VECTORC, "rb");
    assert(fO != NULL);
    size_t ret = fread(orig, sizeof(float), vecSize, fO);
    assert(ret == vecSize);

    // Alloc output data
    c = (float *)malloc(vecSize * sizeof(float));

    getTime(&s);
    // Alloc & init input data
    assert(cudaMalloc((void **)&a, vecSize * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc((void **)&b, vecSize * sizeof(float)) == cudaSuccess);
    getTime(&t);
    printTime(&s, &t, "CUDA:Alloc: ", "\n");

    error = doTest(a, b, c, orig, "CUDA");

    getTime(&s);
    FILE * fC = fopen("vectorC_cuda", "wb");
    ret = fwrite(c, sizeof(float), vecSize, fC);
    assert(ret == vecSize);
    fclose(fC);
    getTime(&t);
    printTime(&s, &t, "CUDA:Write: ", "\n");

    getTime(&s);
    cudaFree(a);
    cudaFree(b);
    getTime(&t);
    printTime(&s, &t, "CUDA:Free: ", "\n");

    free(c);

    return error != 0.f;
}
