#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac/cuda.h>

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
	float *a, *b;
	float **c = (float **)ptr;
	gmactime_t s, t;
	gmacError_t ret = gmacSuccess;

	getTime(&s);
	// Alloc & init input data
	ret = gmacMalloc((void **)&a, vecSize * sizeof(float));
	assert(ret == gmacSuccess);
	ret = gmacMalloc((void **)&b, vecSize * sizeof(float));
	assert(ret == gmacSuccess);

	// Alloc output data
	ret = gmacMalloc((void **)c, vecSize * sizeof(float));
	assert(ret == gmacSuccess);
	getTime(&t);
	printTime(&s, &t, "Alloc: ", "\n");

    // Init the input data
    getTime(&s);
	valueInit(a, 1.0, vecSize);
	valueInit(b, 1.0, vecSize);
    getTime(&t);
    printTime(&s, &t, "Init: ", "\n");

	// Call the kernel
	dim3 Db(blockSize);
	dim3 Dg((unsigned int)vecSize / blockSize);
	if(vecSize % blockSize) Dg.x++;
	getTime(&s);
	vecAdd<<<Dg, Db>>>(gmacPtr(*c), gmacPtr(a), gmacPtr(b), vecSize);
	if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
	getTime(&t);
	printTime(&s, &t, "Run: ", "\n");

	getTime(&s);
	float error = 0;
	for(unsigned i = 0; i < vecSize; i++) {
		error += (*c)[i] - (a[i] + b[i]);
	}
	getTime(&t);
	printTime(&s, &t, "Check: ", "\n");

    getTime(&s);
	gmacFree(a);
	gmacFree(b);
	gmacFree(*c);
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
