#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cassert>

#include <gmac/cl>

#include "utils.h"


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
	const unsigned vecSize = 32 * 1024 * 1024;

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

    
    // Alloc & init input data
    error_code = cl::malloc(queues[0], (void **)&a, vecSize * sizeof(float));
	assert(error_code == CL_SUCCESS);
    error_code = cl::malloc(queues[0], (void **)&b, vecSize * sizeof(float));
	assert(error_code == CL_SUCCESS);
    // Alloc output data
    error_code = cl::malloc(queues[0], (void **)&c, vecSize * sizeof(float));
	assert(error_code == CL_SUCCESS);
    
    float sum = 0.f;
    for(unsigned i = 0; i < vecSize; i++) {
        a[i] = 1.f * rand() / RAND_MAX;
        b[i] = 1.f * rand() / RAND_MAX;
        sum += a[i] + b[i];
    }
    
    // Call the kernel
    size_t local_size = blockSize;
    size_t global_size = vecSize / blockSize;
    if(vecSize % blockSize) global_size++;
    global_size *= local_size;

    error_code = kernel.setArg(0, cl::getBuffer(contexts[0], c));
	assert(error_code == CL_SUCCESS);
    error_code = kernel.setArg(1, cl::getBuffer(contexts[0], a));
	assert(error_code == CL_SUCCESS);
    error_code = kernel.setArg(2, cl::getBuffer(contexts[0], b));
	assert(error_code == CL_SUCCESS);
    error_code = kernel.setArg(3, vecSize);
	assert(error_code == CL_SUCCESS);

    cl::NDRange offset(0);
    cl::NDRange local(local_size);
    cl::NDRange global(global_size);

    error_code = queues[0].enqueueNDRangeKernel(kernel, offset, global, local);
	assert(error_code == CL_SUCCESS);
    error_code = queues[0].finish();
	assert(error_code == CL_SUCCESS);

    float check = 0.f;
    for(unsigned i = 0; i < vecSize; i++) {
        check += c[i];
    }

    if (sum != check) {
        printf("Sum: %f vs %f\n", sum, check);
        abort();
    }

    cl::free(queues[0], a);
    cl::free(queues[0], b);
    cl::free(queues[0], c);
    return 0;
}
