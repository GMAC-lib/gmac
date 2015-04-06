#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cassert>

#include <gmac/cl.h>

#include "utils.h"

const char *vecSizeStr = "GMAC_VECSIZE";
const unsigned vecSizeDefault = 1 * 1024 * 1024;
unsigned vecSize = vecSizeDefault;

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
    cl_platform_id platform;
    cl_device_id device;
    cl_int error_code;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;
	float *a, *b, *c;

	fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

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
    error_code = clMalloc(command_queue, (void **)&a, vecSize * sizeof(float));
	assert(error_code == CL_SUCCESS);
    error_code = clMalloc(command_queue, (void **)&b, vecSize * sizeof(float));
	assert(error_code == CL_SUCCESS);
    // Alloc output data
    error_code = clMalloc(command_queue, (void **)&c, vecSize * sizeof(float));
	assert(error_code == CL_SUCCESS);
    
    float sum = 0.f;
   
    for(unsigned i = 0; i < vecSize; i++) {
        a[i] = 1.f * rand() / RAND_MAX;
        b[i] = 1.f * rand() / RAND_MAX;
        sum += a[i] + b[i];
    }
    
    // Call the kernel    
    size_t global_size = vecSize;

    cl_mem c_device = clGetBuffer(context, c);
    error_code = clSetKernelArg(kernel, 0, sizeof(cl_mem), &c_device);
	assert(error_code == CL_SUCCESS);
    cl_mem a_device = clGetBuffer(context, a);
    error_code = clSetKernelArg(kernel, 1, sizeof(cl_mem), &a_device);
	assert(error_code == CL_SUCCESS);
    cl_mem b_device = clGetBuffer(context, b);
    error_code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &b_device);
	assert(error_code == CL_SUCCESS);
    error_code = clSetKernelArg(kernel, 3, sizeof(vecSize), &vecSize);
	assert(error_code == CL_SUCCESS);

    error_code = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
	assert(error_code == CL_SUCCESS);
    error_code = clFinish(command_queue);
	assert(error_code == CL_SUCCESS);
    
    float error = 0.f;
    float check = 0.f;
    for(unsigned i = 0; i < vecSize; i++) {
        error += c[i] - (a[i] + b[i]);
        check += c[i];
    }
    
    fprintf(stderr, "Error: %f\n", error);

    if (sum != check) {
        printf("Sum: %f vs %f\n", sum, check);
        abort();
    }

	/* Clean up resources */
    clFree(command_queue, a);
    clFree(command_queue, b);
    clFree(command_queue, c);

	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	
    return 0;
}
