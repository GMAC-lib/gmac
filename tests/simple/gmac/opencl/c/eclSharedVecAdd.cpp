#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <gmac/opencl.h>

#include "utils.h"
#include "debug.h"

const char *nIterStr = "GMAC_NITER";
const char *vecSizeStr = "GMAC_VECSIZE";

const unsigned nIterDefault = 2;
const size_t vecSizeDefault = 1024 * 1024;

unsigned nIter = 0;
unsigned vecSize = 0;
const size_t blockSize = 256;


static float *a, *b;
static struct param {
	int i;
	float *ptr;
	char *prefix;
} *param;

const char *kernel = "\
					 __kernel void vecAdd(__global float *c, __global const float *a, __global const float *b, unsigned size, unsigned offset)\
					 {\
					 unsigned i = get_global_id(0);\
					 if(i >= size) return;\
					 \
					 c[i] = a[i + offset] + b[i + offset];\
					 }\
					 ";

void *addVector(void *ptr)
{
	static char buffer[1024];

	gmactime_t s, t;
	struct param *p = (struct param *)ptr;
	char *prefix = p->prefix;
	ecl_error ret = eclSuccess;

	ret = eclMalloc((void **)&p->ptr, vecSize * sizeof(float));
	assert(ret == eclSuccess);

	// Call the kernel
	size_t localSize = blockSize;
	size_t globalSize = vecSize / blockSize;
	if(vecSize % blockSize) globalSize++;
	globalSize *= localSize;

	getTime(&s);
	ecl_kernel kernel;

	assert(eclGetKernel("vecAdd", &kernel) == eclSuccess);
	assert(eclSetKernelArgPtr(kernel, 0, p->ptr) == eclSuccess);
	assert(eclSetKernelArgPtr(kernel, 1, a) == eclSuccess);
	assert(eclSetKernelArgPtr(kernel, 2, b) == eclSuccess);
	assert(eclSetKernelArg(kernel, 3, sizeof(vecSize), &vecSize) == eclSuccess);
	unsigned offset = p->i * long(vecSize);
	assert(eclSetKernelArg(kernel, 4, sizeof(offset), &offset) == eclSuccess);
	assert(eclCallNDRange(kernel, 1, NULL, &globalSize, &localSize) == eclSuccess);

	getTime(&t);
	snprintf(buffer, 1024, "%s-Run: ", prefix);
	printTime(&s, &t, buffer, "\n");

	getTime(&s);
	float error = 0;
	for(unsigned i = 0; i < vecSize; i++) {
		error += p->ptr[i] - (a[i + p->i * vecSize] + b[i + p->i * vecSize]);
		//error += p->ptr[i] - 1.0f;
	}

	eclReleaseKernel(kernel);

	getTime(&t);
	snprintf(buffer, 1024, "%s-CheckFull: ", prefix);
	printTime(&s, &t, buffer, "\n");
	assert(error == 0);

	return NULL;
}

float do_test(GmacGlobalMallocType allocType, const char *prefix)
{
	static char buffer[1024];
	thread_t *nThread;
	unsigned n = 0;
	ecl_error ret = eclSuccess;

	gmactime_t s, t;

	nThread = (thread_t *)malloc(nIter * sizeof(thread_t));
	param = (struct param *)malloc(nIter * sizeof(struct param));

	getTime(&s);
	// Alloc & init input data
	ret = eclGlobalMalloc((void **)&a, nIter * vecSize * sizeof(float), allocType);
	assert(ret == eclSuccess);
	ret = eclGlobalMalloc((void **)&b, nIter * vecSize * sizeof(float), allocType);
	assert(ret == eclSuccess);

	// Alloc output data
	getTime(&t);
	snprintf(buffer, 1024, "%s-Alloc: ", prefix);
	printTime(&s, &t, buffer, "\n");

	getTime(&s);
	valueInit(a, 1.0, nIter * vecSize);
	valueInit(b, 1.0, nIter * vecSize);
	getTime(&t);

	snprintf(buffer, 1024, "%s-Init: ", prefix);
	printTime(&s, &t, buffer, "\n");

	for(n = 0; n < nIter; n++) {
		param[n].i = n;
		param[n].prefix = (char *) prefix;
		nThread[n] = thread_create(addVector, &(param[n]));
	}

	for(n = 0; n < nIter; n++) {
		thread_wait(nThread[n]);
	}

	getTime(&s);
	float error = 0;
	for(n = 0; n < nIter; n++) {
		for(unsigned i = 0; i < vecSize; i++) {
			error += param[n].ptr[i] - 2.f;
		}
	}
	getTime(&t);

	snprintf(buffer, 1024, "%s-Check: ", prefix);
	printTime(&s, &t, buffer, "\n");

	getTime(&s);
	for(n = 0; n < nIter; n++) {
		eclFree(param[n].ptr);
	}

	eclFree(a);
	eclFree(b);

	free(param);
	free(nThread);

	getTime(&t);

	snprintf(buffer, 1024, "%s-Free: ", prefix);
	printTime(&s, &t, buffer, "\n");

	return error;
}


int main(int argc, char *argv[])
{
	assert(eclCompileSource(kernel) == eclSuccess);

	setParam<unsigned>(&nIter, nIterStr, nIterDefault);
	setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);

	vecSize = vecSize / nIter;
	if(vecSize % nIter) vecSize++;

	float error;

	error = do_test(GMAC_GLOBAL_MALLOC_REPLICATED, "Replicated");
	if (error != 0.f) abort();
#if 0
	error = do_test(GMAC_GLOBAL_MALLOC_CENTRALIZED, "Centralized");
	if (error != 0.f) abort();
#endif

	return 0;
}
