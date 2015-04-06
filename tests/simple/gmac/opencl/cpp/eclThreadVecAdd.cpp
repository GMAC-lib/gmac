#include <cstdio>
#include <cstdlib>
#include <time.h>

#include <gmac/opencl>
#include <iostream>
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
	ecl::error ret;
	getTime(&s);
	// Alloc & init input data
	ret = ecl::malloc((void **)&a, vecSize * sizeof(float));
	assert(ret == eclSuccess);
	ret = ecl::malloc((void **)&b, vecSize * sizeof(float));
	assert(ret == eclSuccess);

	for(unsigned i = 0; i < vecSize; i++) {
		a[i] = 1.f * rand() / RAND_MAX;
		b[i] = 1.f * rand() / RAND_MAX;
	}

	// Alloc output data
	ret = ecl::malloc((void **)c, vecSize * sizeof(float));
	assert(ret == eclSuccess);
	getTime(&t);
	printTime(&s, &t, "Alloc: ", "\n");

	// Call the kernel
	getTime(&s);
	ecl::config localSize(blockSize);
	ecl::config globalSize(vecSize / blockSize);
	if(vecSize % blockSize) globalSize.x++;
	globalSize.x *= localSize.x;

	ecl::kernel kernel("vecAdd", ret);
	assert(ret == eclSuccess);
	ret = kernel.setArg(0, *c);
	assert(ret == eclSuccess);
	ret = kernel.setArg(1, a);
	assert(ret == eclSuccess);
	ret = kernel.setArg(2, b);
	assert(ret == eclSuccess);
	ret = kernel.setArg(3, vecSize);
	assert(ret == eclSuccess);

	ret = kernel.callNDRange(globalSize, localSize);
	assert(ret == eclSuccess);

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

	ecl::free(a);
	ecl::free(b);
	ecl::free(*c);

	return NULL;
}

int main(int argc, char *argv[])
{
	thread_t *nThread;
	unsigned n = 0;
	gmactime_t st, en;

	setParam<unsigned>(&nIter, nIterStr, nIterDefault);
	setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);

	assert(ecl::compileSource(kernel) == eclSuccess);

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
