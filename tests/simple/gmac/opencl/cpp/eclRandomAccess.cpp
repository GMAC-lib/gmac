#include <cstdio>
#include <cstdlib>
#include <time.h>

#include <gmac/opencl>

#include "utils.h"
#include "debug.h"

const char *vecSizeStr = "GMAC_VECSIZE";
const unsigned vecSizeDefault = 16 * 1024 * 1024;
unsigned vecSize = vecSizeDefault;

const char *operationsStr = "GMAC_OPERATIONS";
const unsigned operationsDefault = 256 * 1024;
unsigned operations = operationsDefault;

const char *msg = "Done!";

const char *kernel = "\
					 __kernel void null(__global float *a)\
					 {\
					 }\
					 ";


int main(int argc, char *argv[])
{
	float *a;
	gmactime_t s, t;

	assert(ecl::compileSource(kernel) == eclSuccess);

	setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);
	fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

	getTime(&s);
	// Alloc data
	a = new (ecl::allocator) float[vecSize];

	assert(a != NULL);

	getTime(&t);
	printTime(&s, &t, "Alloc: ", "\n");

	// Init input data
	getTime(&s);
	ecl::memset(a, 0, vecSize * sizeof(float));

	for(unsigned i = 0; i < operations; i++) {
		double rnd = (double(rand()) / RAND_MAX);
		unsigned pos = unsigned(rnd * (vecSize - 1));
		a[pos]++;
	}

	getTime(&t);
	printTime(&s, &t, "Init: ", "\n");

	// Call the kernel
	getTime(&s);
	ecl::config globalSize(vecSize);

	ecl::error err;
	ecl::kernel kernel("null", err);
	assert(err == eclSuccess);
#ifndef __GXX_EXPERIMENTAL_CXX0X__
	assert(kernel.setArg(0, a) == eclSuccess);
	assert(kernel.callNDRange(globalSize) == eclSuccess);
#else
	assert(kernel(globalSize)(a) == eclSuccess);
#endif

	getTime(&t);
	printTime(&s, &t, "Run: ", "\n");

	getTime(&s);
	float sum = 0.f;
	for(unsigned i = 0; i < vecSize; i++) {
		sum += a[i];
	}
	getTime(&t);
	printTime(&s, &t, "Check: ", "\n");
	fprintf(stderr, "Error: %f\n", sum - float(operations));

	operator delete(a, ecl::allocator);

	return sum != float(operations);
}
