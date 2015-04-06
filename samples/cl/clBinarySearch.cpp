#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cassert>

#include <gmac/cl.h>

#include "utils.h"

#include "clBinarySearchKernel.cl"

int
binarySearchCPUReference(cl_uint *input, cl_uint *output, cl_uint findMe, unsigned vecSize)
{
    cl_uint globalLowerBound = output[0];
    cl_uint isElementFound = output[2];

    if(isElementFound) {
        if(input[globalLowerBound] == findMe)
            return 1;
        else
            return 0;
    } else {
        for(cl_uint i = 0; i < vecSize; i++) {
            if(input[i] == findMe)
                return 0;
        }
        return 1;
    }
}

int main(int argc, char *argv[])
{
    cl_platform_id platform;
    cl_device_id device;
    cl_int error_code;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;

    cl_uint *input, *output;
    cl_uint findMe = 20000;
    unsigned vecSize = 512;
    unsigned int numSubdivisions = 256;

    error_code = clGetPlatformIDs(1, &platform, NULL);
    assert(error_code == CL_SUCCESS);
    error_code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    assert(error_code == CL_SUCCESS);
    context = clCreateContext(0, 1, &device, NULL, NULL, &error_code);
    assert(error_code == CL_SUCCESS);
    command_queue = clCreateCommandQueue(context, device, 0, &error_code);
    assert(error_code == CL_SUCCESS);
    program = clCreateProgramWithSource(context, 1, &code, NULL, &error_code);
    assert(error_code == CL_SUCCESS);
    error_code = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    assert(error_code == CL_SUCCESS);
    kernel = clCreateKernel(program, "binarySearch", &error_code);
    assert(error_code == CL_SUCCESS);

    /* Alloc data */
    error_code = clMalloc(command_queue, (void **)&input, vecSize * sizeof(cl_uint));
	assert(error_code == CL_SUCCESS);
    error_code = clMalloc(command_queue, (void **)&output, sizeof(cl_uint4));
	assert(error_code == CL_SUCCESS);

    /* Random initialisation of input */
    cl_uint max = vecSize * 20;
    input[0] = 0;
    for(cl_uint i = 1; i < vecSize; i++)
        input[i] = input[i-1] + (cl_uint) (max * rand()/(float)RAND_MAX);
  
    /* Print the input data */
    printf("Sorted data: ");
    for(cl_uint i = 0; i < vecSize; i++)
        printf("%d ", input[i]);
 
    /* Call the kernel */
    size_t globalThreads[1];
    size_t localThreads[1];
    localThreads[0] = 256;

    numSubdivisions = vecSize / (cl_uint)localThreads[0];
    if(numSubdivisions < localThreads[0]) {
        numSubdivisions = (cl_uint)localThreads[0];
    }
    globalThreads[0] = numSubdivisions;

    cl_uint globalLowerBound = 0;
    cl_uint globalUpperBound = vecSize - 1;
    cl_uint subdivSize = (globalUpperBound - globalLowerBound + 1)/numSubdivisions;
    cl_uint isElementFound = 0;

    if((input[0] > findMe) || (input[vecSize-1] < findMe)) {
        output[0] = 0;
        output[1] = vecSize - 1;
        output[2] = 0;
        printf("Not find %d\n", findMe);

        return 0;
    }

    output[3] = 1;

    cl_mem input_device = clGetBuffer(context, input);
    cl_mem output_device = clGetBuffer(context, output);
    error_code = clSetKernelArg(kernel, 0, sizeof(cl_mem), &output_device);
	assert(error_code == CL_SUCCESS);
    error_code = clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_device);
	assert(error_code == CL_SUCCESS);
    error_code = clSetKernelArg(kernel, 2, sizeof(findMe), &findMe);
	assert(error_code == CL_SUCCESS);
    while(subdivSize > 1 && output[3] != 0) {
        output[3] = 0;
        error_code = clSetKernelArg(kernel, 3, sizeof(globalLowerBound), &globalLowerBound);
		assert(error_code == CL_SUCCESS);
        error_code = clSetKernelArg(kernel, 4, sizeof(globalUpperBound), &globalUpperBound);
		assert(error_code == CL_SUCCESS);
        error_code = clSetKernelArg(kernel, 5, sizeof(subdivSize), &subdivSize);
		assert(error_code == CL_SUCCESS);

        error_code = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, globalThreads, localThreads, 0, NULL, NULL);
		assert(error_code == CL_SUCCESS);
        error_code = clFinish(command_queue);
		assert(error_code == CL_SUCCESS);

        globalLowerBound = output[0];
        globalUpperBound = output[1];
        subdivSize = (globalUpperBound - globalLowerBound + 1)/numSubdivisions;
    }

    for(cl_uint i = globalLowerBound; i <= globalUpperBound; i++) {
        if(input[i] == findMe) {
            output[0] = i;
            output[1] = i+1;
            output[2] = 1;
        }
    }
    if(output[2] != 1)
        output[2] = 0;

    globalLowerBound = output[0];
    globalUpperBound = output[1];
    isElementFound = output[2];

    printf("\n l = %d, u = %d, isfound = %d, fm = %d\n", globalLowerBound, globalUpperBound, isElementFound, findMe);

    cl_int verified = binarySearchCPUReference(input, output, findMe, vecSize);
    /* Compare the results and see if they match */
    if(verified) {
        printf("Passed!\n");
    } else {
        printf("Failed\n");
    }

	/* Release memory */
    error_code = clFree(command_queue, input);
	assert(error_code == CL_SUCCESS);
    error_code = clFree(command_queue, output);
	assert(error_code == CL_SUCCESS);

	/* Release OpenCL resources */
	error_code = clReleaseKernel(kernel);
	assert(error_code == CL_SUCCESS);
	error_code = clReleaseProgram(program);
	assert(error_code == CL_SUCCESS);
	error_code = clReleaseCommandQueue(command_queue);
	assert(error_code == CL_SUCCESS);
	error_code = clReleaseContext(context);
	assert(error_code == CL_SUCCESS);

    return 0;
}







