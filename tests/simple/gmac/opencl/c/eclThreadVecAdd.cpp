#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac/opencl.h>

#include "utils.h"
#include "debug.h"

const char *nIterStr = "GMAC_NITER";
const char *vecSizeStr = "GMAC_VECSIZE";

const unsigned nIterDefault = 2;
const unsigned vecSizeDefault = 32 * 1024 * 1024;

unsigned nIter = 0;
unsigned vecSize = 0;
const size_t blockSize = 32;

static float **s;

const char *kernel = "\
					 __kernel void vecAdd(__global float *c, __global const float *a, __global const float *b, unsigned size)\
					 {\
					 unsigned i = get_global_id(0);\
					 if(i >= size) return;\
					 \
					 c[i] = a[i] + b[i];\
					 }\
					 ";

void *addVector(void *ptr)
{
	float *a, *b;
	float **c = (float **)ptr;
	gmactime_t s, t;
	ecl_error ret = eclSuccess;

	getTime(&s);
	// Alloc & init input data
	ret = eclMalloc((void **)&a, vecSize * sizeof(float));
	assert(ret == eclSuccess);
	valueInit(a, 1.0, vecSize);
	ret = eclMalloc((void **)&b, vecSize * sizeof(float));
	assert(ret == eclSuccess);
	valueInit(b, 1.0, vecSize);

	// Alloc output data
	ret = eclMalloc((void **)c, vecSize * sizeof(float));
	assert(ret == eclSuccess);
	getTime(&t);
	printTime(&s, &t, "Alloc: ", "\n");

	// Call the kernel
	getTime(&s);
	size_t localSize = blockSize;
	size_t globalSize = vecSize / blockSize;
	if(vecSize % blockSize) globalSize++;
	globalSize *= localSize;

	ecl_kernel kernel;

	assert(eclGetKernel("vecAdd", &kernel) == eclSuccess);

	assert(eclSetKernelArgPtr(kernel, 0, *c) == eclSuccess);
	assert(eclSetKernelArgPtr(kernel, 1, a) == eclSuccess);
	assert(eclSetKernelArgPtr(kernel, 2, b) == eclSuccess);
	assert(eclSetKernelArg(kernel, 3, sizeof(vecSize), &vecSize) == eclSuccess);
	assert(eclCallNDRange(kernel, 1, NULL, &globalSize, &localSize) == eclSuccess);
	getTime(&t);
	printTime(&s, &t, "Run: ", "\n");

	getTime(&s);
	float error = 0;
	for(unsigned i = 0; i < vecSize; i++) {
		error += (*c)[i] - (a[i] + b[i]);
	}
	getTime(&t);
	printTime(&s, &t, "Check: ", "\n");
	fprintf(stdout, "Error: %.02f\n", error);

	eclReleaseKernel(kernel);

	eclFree(a);
	eclFree(b);
	eclFree(*c);

	assert(error == 0.f);

	return NULL;
}

int main(int argc, char *argv[])
{
	thread_t *nThread;
	unsigned n = 0;
	gmactime_t st, en;

	setParam<unsigned>(&nIter, nIterStr, nIterDefault);
	setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);

	assert(eclCompileSource(kernel) == eclSuccess);

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
