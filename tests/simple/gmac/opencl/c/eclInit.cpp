#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac/opencl.h>

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
	gmactime_t s, t;

    assert(eclCompileSource(kernel) == eclSuccess);

    setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);

    getTime(&s);
    // Alloc & init input data
    if(eclMalloc((void **)&a, vecSize * sizeof(int)) != eclSuccess)
        CUFATAL();
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    // Call the kernel
    getTime(&s);
    size_t localSize = blockSize;
    size_t globalSize = vecSize / blockSize;
    if(vecSize % blockSize) globalSize++;
    globalSize = globalSize * localSize;
    ecl_kernel kernel;

    assert(eclGetKernel("vecAdd", &kernel) == eclSuccess);

    assert(eclSetKernelArgPtr(kernel, 0, a) == eclSuccess);
    assert(eclSetKernelArg(kernel, 1, sizeof(vecSize), &vecSize) == eclSuccess);

    assert(eclCallNDRange(kernel, 1, 0, &globalSize, &localSize) == eclSuccess);

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

    eclReleaseKernel(kernel);
    eclFree(a);

    return error != 0;
}
