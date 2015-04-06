#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac/opencl>

#include <fstream>

#include "utils.h"
#include "debug.h"

#ifdef _MSC_VER
#define VECTORA "inputset\\vectorA"
#define VECTORB "inputset\\vectorB"
#define VECTORC "inputset\\vectorC"
#else
#define VECTORA "inputset/vectorA"
#define VECTORB "inputset/vectorB"
#define VECTORC "inputset/vectorC"
#endif

const unsigned vecSize = 1024 * 1024;
const unsigned blockSize = 32;

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
	gmactime_t s, t;
	ecl::error err;

	assert(ecl::compileSource(kernel) == eclSuccess);

	float * orig = (float *) malloc(vecSize * sizeof(float));
	std::ifstream o_file(VECTORC);
	o_file.read((char *)orig, vecSize * sizeof(float));
	o_file.close();

	getTime(&s);
	// Alloc & init input data
	assert(ecl::malloc((void **)&a, vecSize * sizeof(float)) == eclSuccess);
	assert(ecl::malloc((void **)&b, vecSize * sizeof(float)) == eclSuccess);
	assert(ecl::malloc((void **)&c, vecSize * sizeof(float)) == eclSuccess);
	getTime(&t);
	printTime(&s, &t, "Alloc: ", "\n");

	std::ifstream a_file(VECTORA);
	std::ifstream b_file(VECTORB);

	getTime(&s);
	a_file.read((char *)a, vecSize * sizeof(float));
	a_file.close();
	b_file.read((char *)b, vecSize * sizeof(float));
	b_file.close();
	getTime(&t);
	printTime(&s, &t, "Init: ", "\n");

	// Call the kernel
	getTime(&s);
	ecl::config localSize (blockSize);
	ecl::config globalSize (vecSize / blockSize);
	if(vecSize % blockSize) globalSize.x++;
	globalSize.x *= localSize.x;

	ecl::kernel kernel("vecAdd", err);
	assert(err == eclSuccess);
#ifndef __GXX_EXPERIMENTAL_CXX0X__
	err = kernel.setArg(0, c);
	assert(err == eclSuccess);
	err = kernel.setArg(1, a);
	assert(err == eclSuccess);
	err = kernel.setArg(2, b);
	assert(err == eclSuccess);
	err = kernel.setArg(3, vecSize);
	assert(err == eclSuccess);
	err = kernel.callNDRange(globalSize, localSize);
	assert(err == eclSuccess);
#else
	assert(kernel(c, a, b, vecSize)(globalSize, localSize) == eclSuccess);
#endif
	getTime(&t);
	printTime(&s, &t, "Run: ", "\n");

	getTime(&s);
	float error = 0.f;
	for(unsigned i = 0; i < vecSize; i++) {
		error += orig[i] - (c[i]);
	}
	getTime(&t);
	printTime(&s, &t, "Check: ", "\n");

	getTime(&s);
	std::ofstream c_file("vectorC_shared");
	c_file.write((char *)c, vecSize * sizeof(float));
	c_file.close();
	getTime(&t);
	printTime(&s, &t, "Write: ", "\n");

	getTime(&s);
	ecl::free(a);
	ecl::free(b);
	ecl::free(c);
	getTime(&t);
	printTime(&s, &t, "Free: ", "\n");

	return error != 0;
}
