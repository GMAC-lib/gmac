#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include <cuda.h>

#include "utils.h"
#include "debug.h"

const char *nIterStr = "GMAC_NITER";
const char *vecSizeStr = "GMAC_VECSIZE";

const unsigned nIterDefault = 2;
const size_t vecSizeDefault = 16 * 1024 * 1024;

unsigned nIter = 0;
size_t vecSize = 0;
const size_t blockSize = 512;

static float **s;

__global__ void vecAdd(float *c, const float *a, const float *b, size_t vecSize)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= vecSize) return;

	c[i] = a[i] + b[i];
}

void *addVector(void *ptr)
{
    float *a_h, *b_h;
	float *a_d, *b_d, *c_d;
	float **c = (float **)ptr;
	gmactime_t s, t;
	cudaError_t ret = cudaSuccess;

	getTime(&s);
	// Alloc & init input data
    assert((a_h = (float *)malloc(vecSize * sizeof(float))) != NULL);
	ret = cudaMalloc((void **)&a_d, vecSize * sizeof(float));
	assert(ret == cudaSuccess);

    assert((b_h = (float *)malloc(vecSize * sizeof(float))) != NULL);
	ret = cudaMalloc((void **)&b_d, vecSize * sizeof(float));
	assert(ret == cudaSuccess);

	// Alloc output data
    assert((*c = (float *)malloc(vecSize * sizeof(float))) != NULL);
	ret = cudaMalloc((void **)&c_d, vecSize * sizeof(float));
	assert(ret == cudaSuccess);
	getTime(&t);
	printTime(&s, &t, "Alloc: ", "\n");

    // Init the input data
    getTime(&s);
	valueInit(a_h, 1.0, vecSize);
    assert(cudaMemcpy(a_d, a_h, vecSize * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
	valueInit(b_h, 1.0, vecSize);
    assert(cudaMemcpy(b_d, b_h, vecSize * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    getTime(&t);
    printTime(&s, &t, "Init: ", "\n");

	// Call the kernel
	dim3 Db(blockSize);
	dim3 Dg((unsigned int)vecSize / blockSize);
	if(vecSize % blockSize) Dg.x++;
	getTime(&s);
	vecAdd<<<Dg, Db>>>(c_d, a_d, b_d, vecSize);
	assert(cudaThreadSynchronize() == cudaSuccess);
	getTime(&t);
	printTime(&s, &t, "Run: ", "\n");

	getTime(&s);
    assert(cudaMemcpy(*c, c_d, vecSize * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
	float error = 0;
	for(unsigned i = 0; i < vecSize; i++) {
		error += (*c)[i] - (a_h[i] + b_h[i]);
	}
	getTime(&t);
	printTime(&s, &t, "Check: ", "\n");

    getTime(&s);
	cudaFree(a_d);
    free(a_h);
	cudaFree(b_d);
    free(b_h);
	cudaFree(c_d);
    free(*c);
    getTime(&t);
    printTime(&s, &t, "Free: ", "\n");

    assert(error == 0.f);

	return NULL;
}

int main(int argc, char *argv[])
{
	thread_t *nThread;
	unsigned n = 0;
	gmactime_t st, en;

	setParam<unsigned>(&nIter, nIterStr, nIterDefault);
	setParam<size_t>(&vecSize, vecSizeStr, vecSizeDefault);

	vecSize = vecSize / nIter;
	if(vecSize % nIter) vecSize++;

	nThread = (thread_t *)malloc(nIter * sizeof(thread_t));
	s = (float **)malloc(nIter * sizeof(float **));

	getTime(&st);
	for(n = 0; n < nIter; n++) {
		nThread[n] = thread_create(addVector, &s[n]);
	}

	for(n = 0; n < nIter; n++) {
		thread_wait(nThread[n]);
	}

	getTime(&en);
	printTime(&st, &en, "Total: ", "\n");

	free(s);
	free(nThread);

    return 0;
}
