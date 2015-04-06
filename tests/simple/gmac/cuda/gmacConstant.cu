#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include <gmac/cuda.h>
#include <cuda.h>

#include "debug.h"
#include "utils.h"

const unsigned width = 16;
const unsigned height = 16;

__constant__ float constant[width * height];

__global__ void vecAdd(float *c, unsigned width, unsigned height)
{
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

	c[y * width + x] = constant[y * width + x];
}


int main(int argc, char *argv[])
{
	float *data, *c;
    gmactime_t s, t;

	data = (float *)malloc(width * height * sizeof(float));
	for(unsigned i = 0; i < width * height; i++)
		data[i] = float(i);
	assert(cudaMemcpyToSymbol(constant, data, width * height * sizeof(float)) == cudaSuccess);

	// Alloc output data
    getTime(&s);
	if(gmacMalloc((void **)&c, width * height * sizeof(float)) != gmacSuccess)
		CUFATAL();
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

	// Call the kernel
	dim3 Db(width, height);
	dim3 Dg(1);
    getTime(&s);
	vecAdd<<<Dg, Db>>>(gmacPtr(c), width, height);
	if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
    getTime(&t);
    printTime(&s, &t, "Run: ", "\n");

    getTime(&s);
	for(unsigned i = 0; i < width * height; i++) {
		if(c[i] == data[i]) continue;
        return -1;
	}
    getTime(&t);
    printTime(&s, &t, "Check: ", "\n");

    getTime(&s);
	gmacFree(c);
    getTime(&t);
    printTime(&s, &t, "Free: ", "\n");

    return 0;
}
