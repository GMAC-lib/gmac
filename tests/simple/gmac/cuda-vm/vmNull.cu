#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac/cuda.h>
#include <gmac/vm.h>

#include "debug.h"
#include "utils.h"

size_t vecSize = 1 * 1024 * 1024;
const size_t blockSize = 512;

const char *msg = "Done!";

__global__ void null()
{
	return;
}

int main(int argc, char *argv[])
{
	float *a, *b, *c;
	gmactime_t s, t;

	const char *vecStr = getenv("VECTORSIZE");
	if(vecStr != NULL) vecSize = atoi(vecStr) * 1024 * 1024;
	fprintf(stderr,"Vector %zdMB\n", vecSize);
	// Alloc & init input data
	if(gmacMalloc((void **)&a, vecSize * sizeof(float)) != gmacSuccess)
		CUFATAL();
	if(gmacMalloc((void **)&b, vecSize * sizeof(float)) != gmacSuccess)
		CUFATAL();
	// Alloc output data
	if(gmacMalloc((void **)&c, vecSize * sizeof(float)) != gmacSuccess)
		CUFATAL();

	getTime(&s);
	randInit(a, vecSize);
	randInit(b, vecSize);

	// Call the kernel
	dim3 Db(blockSize);
	dim3 Dg(unsigned(vecSize / blockSize));
	if(vecSize % blockSize) Db.x++;
	null<<<Dg, Db>>>();
	getTime(&t);
	printTime(&s, &t, "", " ");

	if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();

	getTime(&s);
	float error = 0;
	for (unsigned i = 0; i < vecSize; i++) {
		error += (a[i] - b[i]);
	}
	getTime(&t);
	printTime(&s, &t, "", "\n");

	fprintf(stderr,"Error: %f\n", error);

	gmacFree(a);
	gmacFree(b);
	gmacFree(c);
}
