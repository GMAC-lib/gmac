#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cassert>

#if defined(__APPLE__)
#   include <OpenCL/cl.h>
#else
#   include <CL/cl.h>
#endif

#include "utils.h"
#include "debug.h"


const char *vecSizeStr = "GMAC_VECSIZE";
const unsigned vecSizeDefault = 16 * 1024 * 1024;
unsigned vecSize = 0;

const char *kernel_source = "\
__kernel void vecAdd(__global float *c, __global const float *a, __global const float *b)\
{\
    unsigned i = get_global_id(0);\
\
    c[i] = a[i] + b[i];\
}\
";


int main(int argc, char *argv[])
{
    cl_platform_id platform;
    cl_device_id device;
    cl_int error_code;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;
	float *a, *b, *c;
    cl_mem d_a, d_b, d_c;
    cl_int err;
	gmactime_t s, t, S, T;

	setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);
	fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

    getTime(&s);
    error_code = clGetPlatformIDs(1, &platform, NULL);
    assert(error_code == CL_SUCCESS);
    error_code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    assert(error_code == CL_SUCCESS);
    context = clCreateContext(0, 1, &device, NULL, NULL, &error_code);
    assert(error_code == CL_SUCCESS);
    command_queue = clCreateCommandQueue(context, device, 0, &error_code);
    assert(error_code == CL_SUCCESS);
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &error_code);
    assert(error_code == CL_SUCCESS);
    error_code = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    assert(error_code == CL_SUCCESS);
    kernel = clCreateKernel(program, "vecAdd", &error_code);
    assert(error_code == CL_SUCCESS);

    // Alloc & init input data
    assert((a = (float *)malloc(vecSize * sizeof(float))) != NULL);
    d_a = clCreateBuffer(context, CL_MEM_READ_WRITE, vecSize * sizeof(float), NULL, &err);
    assert(err == CL_SUCCESS);
    assert((b = (float *)malloc(vecSize * sizeof(float))) != NULL);
    d_b = clCreateBuffer(context, CL_MEM_READ_WRITE, vecSize * sizeof(float), NULL, &err);
    assert(err == CL_SUCCESS);
    // Alloc output data
    assert((c = (float *)malloc(vecSize * sizeof(float))) != NULL);
    d_c = clCreateBuffer(context, CL_MEM_READ_WRITE, vecSize * sizeof(float), NULL, &err);
    assert(err == CL_SUCCESS);
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    getTime(&S);
    getTime(&s);
    randInitMax(a, 1.f, vecSize);
    randInitMax(b, 1.f, vecSize);
    getTime(&t);
    printTime(&s, &t, "Init: ", "\n");

    float sum = 0.f;
    for(unsigned i = 0; i < vecSize; i++) {
        sum += a[i] + b[i];
    }

    // Call the kernel
    getTime(&s);
    size_t global_size = vecSize;

    assert(clEnqueueWriteBuffer(command_queue, d_a, CL_TRUE, 0, vecSize * sizeof(float), a, 0, NULL, NULL) == CL_SUCCESS);
    assert(clEnqueueWriteBuffer(command_queue, d_b, CL_TRUE, 0, vecSize * sizeof(float), b, 0, NULL, NULL) == CL_SUCCESS);

    assert(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_c) == CL_SUCCESS);
    assert(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_a) == CL_SUCCESS);
    assert(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_b) == CL_SUCCESS);

    assert(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL) == CL_SUCCESS);
    assert(clFinish(command_queue) == CL_SUCCESS);

    getTime(&t);
    printTime(&s, &t, "Run: ", "\n");


    getTime(&s);
    assert(clEnqueueReadBuffer(command_queue, d_c, CL_TRUE, 0, vecSize * sizeof(float), c, 0, NULL, NULL) == CL_SUCCESS);
    float check = 0.f;
    for(unsigned i = 0; i < vecSize; i++) {
        check += c[i];
    }
    getTime(&t);
    getTime(&T);
    printTime(&s, &t, "Check: ", "\n");
    fprintf(stderr, "Error: %f\n", fabsf(sum - check));
    printTime(&S, &T, "Total: ", "\n");

    clReleaseMemObject(d_a); free(a);
    clReleaseMemObject(d_b); free(b);
    clReleaseMemObject(d_c); free(c);
    return 0;
}
