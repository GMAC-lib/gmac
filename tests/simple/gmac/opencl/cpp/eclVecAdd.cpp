#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac/opencl>

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


int main(int argc, char *argv[])
{
	float *a, *b, *c;
	gmactime_t s, t, S, T;

	assert(ecl::compileSource(kernel) == eclSuccess);

	setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);
	fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

	getTime(&s);
	// Alloc input data
	a = new (ecl::allocator) float[vecSize];
	b = new (ecl::allocator) float[vecSize];
	// Alloc output data
	c = new (ecl::allocator) float[vecSize];

	assert(a != NULL);
	assert(b != NULL);
	assert(c != NULL);

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

	// Call the kernel
	getTime(&s);
	ecl::config globalSize(vecSize);

	ecl::error err;
	ecl::kernel kernel("vecAdd", err);
	assert(err == eclSuccess);
#ifndef __GXX_EXPERIMENTAL_CXX0X__
	assert(kernel.setArg(0, c) == eclSuccess);
	assert(kernel.setArg(1, a) == eclSuccess);
	assert(kernel.setArg(2, b) == eclSuccess);
	assert(kernel.setArg(3, vecSize) == eclSuccess);
	assert(kernel.callNDRange(globalSize) == eclSuccess);
#else
	assert(kernel(globalSize)(c, a, b, vecSize) == eclSuccess);
#endif

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

	operator delete(a, ecl::allocator);
	operator delete(b, ecl::allocator);
	operator delete(c, ecl::allocator);

	return sum != check;
}
