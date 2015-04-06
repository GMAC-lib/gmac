#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac/opencl.h>

#include "utils.h"
#include "debug.h"

const char *xSizeStr = "GMAC_XSIZE";
const unsigned xSizeDefault = 1024;
unsigned xSize = xSizeDefault;

const char *ySizeStr = "GMAC_YSIZE";
const unsigned ySizeDefault = 128;
unsigned ySize = ySizeDefault;

const char *msg = "Done!";

const char *kernel = "\
					 __kernel void null(__global float *a)\
					 {\
					 }\
					 ";


int main(int argc, char *argv[])
{
	float *a;
	ecl_error ret;
	gmactime_t s, t;

	assert(eclCompileSource(kernel) == eclSuccess);

	setParam<unsigned>(&xSize, xSizeStr, xSizeDefault);
	setParam<unsigned>(&ySize, ySizeStr, ySizeDefault);

	fprintf(stdout, "Matrix: %u x %u\n", xSize , ySize);

	getTime(&s);
	// Alloc data
	ret = eclMalloc((void **)&a, xSize * ySize * sizeof(float));
	assert(ret == eclSuccess);

	getTime(&t);
	printTime(&s, &t, "Alloc: ", "\n");

	// Init input data
	getTime(&s);
	eclMemset(a, 0, xSize * ySize * sizeof(float));

	for (unsigned j = 0; j < xSize; j++) {
		for (unsigned i = 0; i < ySize; i++) {
			a[j + i * xSize] = 1.f;
		}
	}

	getTime(&t);
	printTime(&s, &t, "Init: ", "\n");

	// Call the kernel
	getTime(&s);
	size_t globalSize = 1;
	ecl_kernel kernel;
	ret = eclGetKernel("null", &kernel);
	assert(ret == eclSuccess);
#ifndef __GXX_EXPERIMENTAL_CXX0X__
	ret = eclSetKernelArgPtr(kernel, 0, a);
	assert(ret == eclSuccess);
	ret = eclCallNDRange(kernel, 1, NULL, &globalSize, NULL);
	assert(ret == eclSuccess);
#else
	assert(kernel(1)(a) == eclSuccess);
#endif

	getTime(&t);
	printTime(&s, &t, "Run: ", "\n");

	getTime(&s);
	double sum = 0.f;
	for (unsigned i = 0; i < ySize; i++) {
		for (unsigned j = 0; j < xSize; j++) {
			sum += a[j + i * xSize];
		}
	}
	getTime(&t);
	printTime(&s, &t, "Check: ", "\n");
	fprintf(stderr, "Error: %f\n", sum - double(xSize * ySize));

	eclReleaseKernel(kernel);
	eclFree(a);

	return sum != double(xSize * ySize);
}
