#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac/opencl.h>

#include "utils.h"
#include "barrier.h"
#include "debug.h"

const char *nIterStr = "GMAC_NITER";
const char *vecSizeStr = "GMAC_VECSIZE";

const unsigned nIterDefault = 2;
const unsigned vecSizeDefault = 1024 * 1024;

unsigned nIter = 0;
unsigned vecSize = 0;

const char *kernel = "\
					 __kernel void accum(__global float *c)\
					 {\
					 unsigned i = get_global_id(0);\
					 \
					 c[i] = 0.0f;\
					 for(int n = 0; n < 512 * 1024; n++)\
					 c[i] += 0.01f;\
					 }\
					 ";

float *a = NULL;
barrier_t barr;

void *check(void *ptr)
{
	unsigned n, m, *id = (unsigned *)ptr;
	unsigned pitch = vecSize / nIter;
	for(n = 0; n < 32; n++) {
		// Wait for the main thread to execute the kernel
		barrier_wait(&barr);

		// Do some checks
		for(m = 0; m < pitch; m++) {
			if((pitch + m) >= vecSize) break;
			assert(fabsf(a[(*id * pitch) + m] / (512 * 1024 * 0.01f)) > 0.99);
			a[(*id * pitch) + m] = 0.0f;
		}
		// Wait for everybody to finish
		barrier_wait(&barr);
	}
	return NULL;
}

int main(int argc, char *argv[])
{
	thread_t *nThread;
	unsigned n = 0;
	gmactime_t st, en;

	setParam<unsigned>(&nIter, nIterStr, nIterDefault);
	setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);

	barrier_init(&barr, nIter + 1);

	assert(eclCompileSource(kernel) == eclSuccess);

	nThread = (thread_t *)malloc(nIter * sizeof(thread_t));
	unsigned *ids = (unsigned *)malloc(nIter * sizeof(unsigned));

	getTime(&st);
	for(n = 0; n < nIter; n++) {
		ids[n] = n;
		nThread[n] = thread_create(check, &ids[n]);
	}

	// Alloc & init input data
	ecl_error ret = eclMalloc((void **)&a, vecSize * sizeof(float));
	assert(ret == eclSuccess);

	for(n = 0; n < 32; n++) {
		// Call the kernel
		size_t globalSize = vecSize;
		ecl_kernel kernel;
		assert(eclGetKernel("accum", &kernel) == eclSuccess);
		assert(eclSetKernelArgPtr(kernel, 0, a) == eclSuccess);
		assert(eclCallNDRange(kernel, 1, NULL, &globalSize, NULL) == eclSuccess);
		barrier_wait(&barr);

		// Wait for the threads to do stuff
		barrier_wait(&barr);
	}

	for(n = 0; n < nIter; n++) {
		thread_wait(nThread[n]);
	}

	getTime(&en);
	printTime(&st, &en, "Total: ", "\n");

	free(ids);
	free(nThread);

	return 0;
}
