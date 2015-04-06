#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>

#include "utils.h"
#include "debug.h"


const char *vecSizeStr = "GMAC_VECSIZE";
const size_t vecSizeDefault = 16 * 1024 * 1024;

size_t vecSize = 0;
const size_t blockSize = 512;

__global__ void vecAdd(float *c, float *a, float *b, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= size) return;

    c[i] = a[i] + b[i];
}

void init(float *ptr, unsigned s, float v)
{
    for(unsigned i = 0; i < s; i++) {
        ptr[i] = v;
    }
}


int main(int argc, char *argv[])
{
	float *a, *b, *c;
    float *d_a, *d_b, *d_c;
	gmactime_t s, t, begin, end;

	setParam<size_t>(&vecSize, vecSizeStr, vecSizeDefault);

    getTime(&s);
    // Alloc & init input data
    assert((a = (float *)malloc(vecSize * sizeof(float))) != NULL);
    assert(cudaMalloc((void **)&d_a, vecSize * sizeof(float)) == cudaSuccess);
    assert((b = (float *)malloc(vecSize * sizeof(float))) != NULL);
    assert(cudaMalloc((void **)&d_b, vecSize * sizeof(float)) == cudaSuccess);
    // Alloc output data
    assert((c = (float *)malloc(vecSize * sizeof(float))) != NULL);
    assert(cudaMalloc((void **)&d_c, vecSize * sizeof(float)) == cudaSuccess);
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    float sum = 0.f;

    getTime(&s);
    begin = s;
    randInitMax(a, 1.f, vecSize);
    randInitMax(b, 1.f, vecSize);
    getTime(&t);
    printTime(&s, &t, "Init: ", "\n");

    for(unsigned i = 0; i < vecSize; i++) {
        sum += a[i] + b[i];
    }
    
    // Call the kernel
    getTime(&s);
    assert(cudaMemcpy(d_a, a, vecSize * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(d_b, b, vecSize * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    dim3 Db(blockSize);
    dim3 Dg((unsigned long)vecSize / blockSize);
    if(vecSize % blockSize) Dg.x++;
    vecAdd<<<Dg, Db>>>(d_c, d_a, d_b, vecSize);
    assert(cudaThreadSynchronize() == cudaSuccess);
    getTime(&t);
    printTime(&s, &t, "Run: ", "\n");

    getTime(&s);
    assert(cudaMemcpy(c, d_c, vecSize * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    float check = 0;
    for(unsigned i = 0; i < vecSize; i++) {
        check += c[i];
    }
    getTime(&t);
    end = t;
    printTime(&s, &t, "Check: ", "\n");

    getTime(&s);
    free(a); cudaFree(d_a);
    free(b); cudaFree(d_b);
    free(c); cudaFree(d_c);
    getTime(&t);
    printTime(&s, &t, "Free: ", "\n");
    printTime(&begin, &end, "Total: ", "\n");

    return sum == check;
}
