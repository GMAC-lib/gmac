#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <gmac/opencl.h>

unsigned vecSize = 16 * 1024 * 1024;

const char *kernel = "\
__kernel void vecAdd(__global float *c, __global float *a, __global float *b)\
{\
    unsigned i = get_global_id(0);\
\
    c[i] = a[i] + b[i];\
}\
";


int main(int argc, char *argv[])
{
	float *a, *b, *c;
	ecl_error ret;

    ret = eclCompileSource(kernel);
	assert(ret == eclSuccess);

	fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

    // Alloc & init input data
    ret = eclMalloc((void **)&a, vecSize * sizeof(float));
	assert(ret == eclSuccess);
    ret = eclMalloc((void **)&b, vecSize * sizeof(float));
	assert(ret == eclSuccess);
    // Alloc output data
    ret = eclMalloc((void **)&c, vecSize * sizeof(float));
	assert(ret == eclSuccess);

    for(unsigned i = 0; i < vecSize; i++) {
        a[i] = 1.f * rand() / RAND_MAX;
        b[i] = 1.f * rand() / RAND_MAX;
    }

    // Call the kernel
    size_t globalSize = vecSize;
    ecl_kernel kernel;
    ret = eclGetKernel("vecAdd", &kernel);
	assert(ret == eclSuccess);
    ret = eclSetKernelArgPtr(kernel, 0, c);
	assert(ret == eclSuccess);
    ret = eclSetKernelArgPtr(kernel, 1, a);
	assert(ret == eclSuccess);
    ret = eclSetKernelArgPtr(kernel, 2, b);
	assert(ret == eclSuccess);
    ret = eclCallNDRange(kernel, 1, NULL, &globalSize, NULL);
	assert(ret == eclSuccess);

    // Check the result in the CPU
    float error = 0.f;
    for(unsigned i = 0; i < vecSize; i++) {
        error += c[i] - (a[i] + b[i]);
    }
    fprintf(stderr, "Error: %f\n", error);

    eclReleaseKernel(kernel);

    eclFree(a);
    eclFree(b);
    eclFree(c);

   return 0;
}
