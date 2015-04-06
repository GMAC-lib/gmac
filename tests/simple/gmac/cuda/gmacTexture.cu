#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include <gmac/cuda.h>

#include "debug.h"

const unsigned width = 16;
const unsigned height = 16;

texture<unsigned short, 2, cudaReadModeElementType> tex;

__global__ void vecAdd(unsigned short *c, unsigned width, unsigned height)
{
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

	c[y * width + x] = tex2D(tex, x, y);
}

gmacError_t launchKernel(unsigned short *output, const unsigned short *orig, unsigned width, unsigned height)
{
    gmacError_t ret;

    dim3 Db(width, height);
	dim3 Dg(1);
	vecAdd<<<Dg, Db>>>(gmacPtr(output), width, height);
    ret = gmacThreadSynchronize();
	if(ret != gmacSuccess) CUFATAL();

	for(unsigned i = 0; i < width * height; i++) {
		if(output[i] == orig[i]) continue;
		fprintf(stderr,"Error on %d (%hu)\n", i, output[i]);
		abort();
	}

    return ret;
}

int main(int argc, char *argv[])
{
	unsigned short *data, *c, *d;
	struct cudaArray *array;

    // Initialize input data
	data = (unsigned short *)malloc(width * height * sizeof(unsigned short));
	for(unsigned i = 0; i < width * height; i++)
		data[i] = i;


    ///////////////////////
    // Array-based textures
    ///////////////////////
	assert(cudaMallocArray(&array, &tex.channelDesc, width, height) == cudaSuccess);
	assert(cudaMemcpy2DToArray(array, 0, 0, data, width * sizeof(unsigned short),
			width * sizeof(unsigned short), height, cudaMemcpyHostToDevice) == cudaSuccess);

	assert(cudaBindTextureToArray(tex, array) == cudaSuccess);

	// Alloc output data
	if(gmacMalloc((void **)&c, width * height * sizeof(unsigned short)) != gmacSuccess)
		CUFATAL();

	// Call the kernel
    launchKernel(c, data, width, height);
	

    /////////////////////////
    // Pointer-based textures
    /////////////////////////
    ::memset(c, 0, width * height * sizeof(unsigned short));

    cudaChannelFormatDesc desc = tex.channelDesc;

	// Alloc input data
	if(gmacMalloc((void **)&d, width * height * sizeof(unsigned short)) != gmacSuccess)
		CUFATAL();

	for(unsigned i = 0; i < width * height; i++)
		d[i] = data[i];

    size_t offset;
    cudaBindTexture(&offset, &tex, gmacPtr(d), &desc);
    assert(offset == 0);

	// Call the kernel
    launchKernel(c, data, width, height);
	
    cudaUnbindTexture(&tex);

    // Memory release
    assert(gmacFree(c) == gmacSuccess);
    assert(gmacFree(d) == gmacSuccess);

    assert(cudaFreeArray(array) == cudaSuccess);

    return 0;
}
