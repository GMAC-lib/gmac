#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <gmac/opencl.h>

#include "utils.h"
#include "debug.h"

const char *vecSizeStr = "GMAC_VECSIZE";
const unsigned vecSizeDefault = 16 * 1024 * 1024;
unsigned vecSize = vecSizeDefault;

const char *msg = "Done!";

const char *kernel = "\
					 __kernel void vecAdd(__global float *c, __global const float *a, __global const float *b, unsigned size)\
					 {\
					 unsigned i = get_global_id(0);\
					 if(i >= size) return;\
					 \
					 c[i] = a[i] + b[i];\
					 }\
					 ";

#define STAGES 4

int main(int argc, char *argv[])
{
	float *a, *b, *c;
	gmactime_t s, t, S, T;

	assert(eclCompileSource(kernel) == eclSuccess);

	setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);
	fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

	getTime(&s);
	// Alloc input data
	ecl_error ret;
	ret = eclMalloc((void **)&a, vecSize * sizeof(float));
	assert(ret == eclSuccess);
	ret = eclMalloc((void **)&b, vecSize * sizeof(float));
	assert(ret == eclSuccess);

	// Alloc output data
	ret = eclMalloc((void **)&c, vecSize * sizeof(float));
	assert(ret == eclSuccess);

	getTime(&t);
	printTime(&s, &t, "Alloc: ", "\n");

	getTime(&S);
	getTime(&s);
	randInitMax(a, 10.f, vecSize);
	randInitMax(b, 10.f, vecSize);

	getTime(&t);
	printTime(&s, &t, "Init: ", "\n");

	float sum = 0.f;
	for(unsigned i = 0; i < vecSize; i++) {
		sum += a[i] + b[i];
	}

	getTime(&s);

	unsigned stageSize = vecSize/STAGES;

	// Perform the computation in multiple stages
	for (unsigned i = 0; i < STAGES; i++) {
		size_t globalSize[1] = { vecSize };
		ecl_kernel kernel;
		ret = eclGetKernel("vecAdd", &kernel);
		assert(ret == eclSuccess);
		ret = eclSetKernelArgPtr(kernel, 0, c + i * stageSize);
		assert(ret == eclSuccess);
		ret = eclSetKernelArgPtr(kernel, 1, a + i * stageSize);
		assert(ret == eclSuccess);
		ret = eclSetKernelArgPtr(kernel, 2, b + i * stageSize);
		assert(ret == eclSuccess);
		ret = eclSetKernelArg(kernel, 3, sizeof(unsigned int), &stageSize);
		assert(ret == eclSuccess);
		ret = eclCallNDRange(kernel, 1, NULL, globalSize, NULL);
		assert(ret == eclSuccess);
	}

	getTime(&t);
	printTime(&s, &t, "Run: ", "\n");

	getTime(&s);
	float check = 0.f;
	for(unsigned i = 0; i < vecSize; i++) {
		check += c[i];
	}
	getTime(&t);
	getTime(&T);
	printTime(&s, &t, "Check: ", "\n");
	fprintf(stderr, "Error: %f\n", fabsf(sum - check));
	printTime(&S, &T, "Total: ", "\n");

	eclFree(a);
	eclFree(b);
	eclFree(c);

	return sum != check;
}
