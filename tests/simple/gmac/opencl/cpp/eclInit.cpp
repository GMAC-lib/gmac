#include <cstdio>
#include <cstdlib>
#include <time.h>

#include <gmac/opencl>

#include "utils.h"
#include "debug.h"

const char *vecSizeStr = "GMAC_VECSIZE";
const size_t vecSizeDefault = 16 * 1024 * 1024;

unsigned vecSize = 0;
const unsigned blockSize = 256;

const char *msg = "Done!";
const char *kernel = "\
					 __kernel void vecAdd( __global int *a, unsigned size)\
					 {\
					 int i = get_global_id(0);\
					 if(i >= size) return;\
					 \
					 a[i] = i;\
					 }\
					 ";

int main(int argc, char *argv[])
{
	int *a;
	ecl::error ret;
	gmactime_t s, t;

	assert(ecl::compileSource(kernel) == eclSuccess);

	setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);

	getTime(&s);
	//Alloc & init input data

	assert(ecl::malloc((void **)&a, vecSize * sizeof(int)) == eclSuccess);

	getTime(&t);
	printTime(&s, &t, "Alloc: ", "\n");

	// Call the kernel
	getTime(&s);
	ecl::config localSize(blockSize);
	ecl::config globalSize (vecSize / blockSize);
	if(vecSize % blockSize) globalSize.x++;
	globalSize.x = globalSize.x * localSize.x;

	ecl::kernel kernel("vecAdd", ret);
	assert(ret == eclSuccess);
#ifndef __GXX_EXPERIMENTAL_CXX0X__
	ret = kernel.setArg(0, a);
	assert(ret == eclSuccess);
	ret = kernel.setArg(1, vecSize);
	assert(ret == eclSuccess);
	ret = kernel.callNDRange(globalSize, localSize, 0);
	assert(ret == eclSuccess);
#else
	assert(kernel(a, vecSize)(globalSize, localSize, 0) == eclSuccess);
#endif

	getTime(&t);
	printTime(&s, &t, "Run: ", "\n");

	getTime(&s);
	float error = 0;
	for(unsigned i = 0; i < vecSize; i++) {
		error += 1.0f * (a[i] - i);
	}
	getTime(&t);
	printTime(&s, &t, "Check: ", "\n");

	fprintf(stderr, "Error: %f\n", error);

	ecl::free(a);
	return error != 0;
}
