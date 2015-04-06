#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac/cuda.h>

#include "utils.h"
#include "debug.h"

const char *nIterStr = "GMAC_NITER";
const char *vecSizeStr = "GMAC_VECSIZE";

const unsigned nIterDefault = 1;
const size_t vecSizeDefault = 4 * 1024 * 1024;

unsigned nIter = 0;
size_t vecSize = 0;
const size_t blockSize = 512;


static float *a, *b;
static struct param {
	int i;
	float *ptr;
    char *prefix;
} *param;


__global__ void vecAdd(float *c, float *a, float *b, size_t vecSize)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= vecSize) return;

	c[i] = a[i] + b[i];
}

void *addVector(void *ptr)
{
    static char buffer[1024];

	gmactime_t s, t;
	struct param *p = (struct param *)ptr;
    char *prefix = p->prefix;
	gmacError_t ret = gmacSuccess;

	ret = gmacMalloc((void **)&p->ptr, vecSize * sizeof(float));
	assert(ret == gmacSuccess);

	// Call the kernel
	dim3 Db(blockSize);
	dim3 Dg(int(vecSize / blockSize));
	if(vecSize % blockSize) Dg.x++;

	getTime(&s);
	 vecAdd<<<Dg, Db>>>(gmacPtr((p->ptr)), gmacPtr(a + p->i * vecSize), gmacPtr(b + p->i * vecSize), vecSize);
	if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
	getTime(&t);
    snprintf(buffer, 1024, "%s-Run: ", prefix);
	printTime(&s, &t, buffer, "\n");

	getTime(&s);
	float error = 0;
	for(unsigned i = 0; i < vecSize; i++) {
		error += p->ptr[i] - (a[i + p->i * vecSize] + b[i + p->i * vecSize]);
		//error += (a[i] - b[i]);
	}
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
	gmacError_t ret = gmacSuccess;

	gmactime_t s, t;

	nThread = (thread_t *)malloc(nIter * sizeof(thread_t));
	param = (struct param *)malloc(nIter * sizeof(struct param));

	getTime(&s);
	// Alloc & init input data
	ret = gmacGlobalMalloc((void **)&a, nIter * vecSize * sizeof(float));
	assert(ret == gmacSuccess);
	ret = gmacGlobalMalloc((void **)&b, nIter * vecSize * sizeof(float));
	assert(ret == gmacSuccess);

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
		gmacFree(param[n].ptr);
	}

	gmacFree(a);
	gmacFree(b);

	free(param);
	free(nThread);

    getTime(&t);

    snprintf(buffer, 1024, "%s-Free: ", prefix);
    printTime(&s, &t, buffer, "\n");

    return error;
}

int main(int argc, char *argv[])
{
	setParam<unsigned>(&nIter, nIterStr, nIterDefault);
	setParam<size_t>(&vecSize, vecSizeStr, vecSizeDefault);

	vecSize = vecSize / nIter;
	if(vecSize % nIter) vecSize++;

    float error;

    error = do_test(GMAC_GLOBAL_MALLOC_REPLICATED, "Replicated");
    if (error != 0.f) abort();
    error = do_test(GMAC_GLOBAL_MALLOC_CENTRALIZED, "Centralized");
    if (error != 0.f) abort();
    
    return 0;
}
