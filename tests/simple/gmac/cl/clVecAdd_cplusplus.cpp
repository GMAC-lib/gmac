#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cassert>

#include "gmac/cl"

#include "utils.h"
#include "debug.h"

const char *vecSizeStr = "GMAC_VECSIZE";
const unsigned vecSizeDefault = 32 * 1024 * 1024;
unsigned vecSize = 0;

const size_t blockSize = 32;

const char *msg = "Done!";

const char *kernel_source = "\
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
    cl_int error_code;
	float *a, *b, *c;
	gmactime_t s, t;

	setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);
	fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

    error_code = cl::Helper::init();
    assert(error_code == CL_SUCCESS);

    VECTOR_CLASS<cl::Helper> &platforms = cl::Helper::getPlatforms();
    assert(platforms.size() > 0);

    VECTOR_CLASS<cl::Device> &devices = platforms[0].getDevices();
    assert(devices.size() > 0);

    VECTOR_CLASS<cl::Context> &contexts = platforms[0].getContexts();
    assert(contexts.size() > 0);

    VECTOR_CLASS<cl::CommandQueue> &queues= platforms[0].getCommandQueues();
    assert(queues.size() > 0);

    VECTOR_CLASS<cl::Program> programs = platforms[0].buildProgram(kernel_source, &error_code);
    assert(error_code == CL_SUCCESS);
    assert(programs.size() > 0);

    cl::Kernel kernel(programs[0], "vecAdd", &error_code);
    assert(error_code == CL_SUCCESS);

    getTime(&s);
    // Alloc & init input data
    assert(cl::malloc(contexts[0], (void **)&a, vecSize * sizeof(float)) == CL_SUCCESS);
    assert(cl::malloc(contexts[0], (void **)&b, vecSize * sizeof(float)) == CL_SUCCESS);
    // Alloc output data
    assert(cl::malloc(contexts[0], (void **)&c, vecSize * sizeof(float)) == CL_SUCCESS);
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");


    float sum = 0.f;

    getTime(&s);
    valueInit(a, 1.f, vecSize);
    valueInit(b, 1.f, vecSize);
    getTime(&t);
    printTime(&s, &t, "Init: ", "\n");

    for(unsigned i = 0; i < vecSize; i++) {
        sum += a[i] + b[i];
    }
    
    // Call the kernel
    getTime(&s);
    size_t local_size = blockSize;
    size_t global_size = vecSize / blockSize;
    if(vecSize % blockSize) global_size++;
    global_size *= local_size;

    assert(kernel.setArg(0, cl::getBuffer(contexts[0], c)) == CL_SUCCESS);
    assert(kernel.setArg(1, cl::getBuffer(contexts[0], a)) == CL_SUCCESS);
    assert(kernel.setArg(2, cl::getBuffer(contexts[0], b)) == CL_SUCCESS);
    assert(kernel.setArg(3, vecSize) == CL_SUCCESS);

    cl::NDRange offset(0);
    cl::NDRange local(local_size);
    cl::NDRange global(global_size);

    assert(queues[0].enqueueNDRangeKernel(kernel, offset, global, local) == CL_SUCCESS);

    assert(queues[0].finish() == CL_SUCCESS);

    getTime(&t);
    printTime(&s, &t, "Run: ", "\n");


    getTime(&s);
    float error = 0.f;
    float check = 0.f;
    for(unsigned i = 0; i < vecSize; i++) {
        error += c[i] - (a[i] + b[i]);
        check += c[i];
    }
    getTime(&t);
    printTime(&s, &t, "Check: ", "\n");
    fprintf(stderr, "Error: %f\n", error);

    if (sum != check) {
        printf("Sum: %f vs %f\n", sum, check);
        abort();
    }

    cl::free(contexts[0], a);
    cl::free(contexts[0], b);
    cl::free(contexts[0], c);
    return 0;
}
