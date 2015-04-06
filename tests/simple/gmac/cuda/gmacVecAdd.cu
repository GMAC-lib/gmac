#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac/cuda>

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
	gmactime_t s, t, begin, end;

	setParam<size_t>(&vecSize, vecSizeStr, vecSizeDefault);

    getTime(&s);
    // Alloc input data
    a = new (gmac::allocator) float[vecSize];
    b = new (gmac::allocator) float[vecSize];
    // Alloc output data
    c = new (gmac::allocator) float[vecSize];

    assert(a != NULL);
    assert(b != NULL);
    assert(c != NULL);
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    float sum = 0.f;

    // Init input data
    getTime(&s);
    begin = s;
    randInitMax(a, 10.f, vecSize);
    randInitMax(b, 10.f, vecSize);
    getTime(&t);
    printTime(&s, &t, "Init: ", "\n");

    for(unsigned i = 0; i < vecSize; i++) {
        sum += a[i] + b[i];
    }
    
    // Call the kernel
    getTime(&s);
    dim3 Db(blockSize);
    dim3 Dg((unsigned long)vecSize / blockSize);
    if(vecSize % blockSize) Dg.x++;
    vecAdd<<<Dg, Db>>>(gmac::ptr(c), gmac::ptr(a), gmac::ptr(b), vecSize);
    assert(gmac::threadSynchronize() == gmacSuccess);
    getTime(&t);
    printTime(&s, &t, "Run: ", "\n");

    getTime(&s);
    float check = 0;
    for(unsigned i = 0; i < vecSize; i++) {
        check += c[i];
    }
    getTime(&t);
    end = t;
    printTime(&s, &t, "Check: ", "\n");

    getTime(&s);
    gmac::free(a);
    gmac::free(b);
    gmac::free(c);
    getTime(&t);
    printTime(&s, &t, "Free: ", "\n");
    printTime(&begin, &end, "Total: ", "\n");

    return sum != check;
}
