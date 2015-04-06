#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "gmac/cl.h"

#include "utils.h"
#include "debug.h"

#include "clBitonicSortKernel.cl"

#define GROUP_SIZE 1

void
swapIfFirstIsGreater(cl_uint *a, cl_uint *b)
{
    if(*a > *b) {
        cl_uint temp = *a;
        *a = *b;
        *b = temp;
    }
}

/*
 * sorts the input array (in place) using the bubble sort algorithm
 * sorts in increasing order if sortIncreasing is CL_TRUE
 * else sorts in decreasing order
 * length specifies the length of the array
 */
void
bubbleSortCPUReference(
    cl_uint *input,
    const cl_uint length,
    const cl_bool sortIncreasing)
{ 
	cl_uint i, j;
	for(i = length-1; i > 0; i--) {
		for(j = 0; j < i; j++) {
			if(sortIncreasing)
				swapIfFirstIsGreater(&input[j], &input[j + 1]);
			else
				swapIfFirstIsGreater(&input[j + 1], &input[j]);
		}
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

    gmactime_t s, t;

    cl_uint seed = 123;
    cl_uint sortDescending = 0;
    cl_uint *input = NULL;
    cl_uint *verificationInput;
    cl_uint length = 1024;

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
    kernel = clCreateKernel(program, "bitonicSort", &error_code);
    assert(error_code == CL_SUCCESS);

    getTime(&s);
    // Alloc
    error_code = clMalloc(command_queue, (void **)&input, length * sizeof(cl_uint));
	assert(error_code == CL_SUCCESS);
    verificationInput = (cl_uint *) malloc(length*sizeof(cl_uint));
    if(verificationInput == NULL)
        return 0;
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    getTime(&s);
    /* random initialisation of input */
    srand(seed);
    const cl_uint rangeMin = 0;
    const cl_uint rangeMax = 255;
    double range = double(rangeMax - rangeMin) + 1.0;
    for(cl_uint i = 0; i < length; i++) {
        input[i] = rangeMin + (cl_uint)(range * rand() / (RAND_MAX + 1.0));
    }
    memcpy(verificationInput, input, length*sizeof(cl_uint));
    getTime(&t);
    printTime(&s, &t, "Init: ", "\n");

    getTime(&s);
    // Print the input data
    printf("Unsorted Input: ");
    for(cl_uint i = 0; i < length; i++)
        printf("%d ", input[i]);
    getTime(&t);
    printTime(&s, &t, "\nPrint: ", "\n");

    getTime(&s);
    cl_uint numStages = 0;
    size_t globalThreads[1] = {length / 2};
    size_t localThreads[1] = {GROUP_SIZE};

    for(cl_uint temp = length; temp > 1; temp >>= 1)
        ++numStages;
    cl_mem input_device = clGetBuffer(context, input);
    error_code = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_device);
	assert(error_code == CL_SUCCESS);
    error_code = clSetKernelArg(kernel, 3, sizeof(length), &length);
	assert(error_code == CL_SUCCESS);
    error_code = clSetKernelArg(kernel, 4, sizeof(sortDescending), &sortDescending);
	assert(error_code == CL_SUCCESS);
    for(cl_uint stage = 0; stage < numStages; ++stage) {
        /* stage of the algorithm */
        error_code = clSetKernelArg(kernel, 1, sizeof(stage), &stage);
		assert(error_code == CL_SUCCESS);
        /* Every stage has stage+1 passes. */
        for(cl_uint passOfStage = 0; passOfStage < stage + 1; ++passOfStage) {
            /* pass of the current stage */
            error_code = clSetKernelArg(kernel, 2, sizeof(passOfStage), &passOfStage);
			assert(error_code == CL_SUCCESS);
            error_code = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, globalThreads, localThreads, 0, NULL, NULL);
			assert(error_code == CL_SUCCESS);
            error_code = clFinish(command_queue);
			assert(error_code == CL_SUCCESS);
        }
    }
    getTime(&t);
    printTime(&s, &t, "Run: ", "\n");

    printf("Output: ");
    for(cl_uint i = 0; i < length; i++) {
        printf("%d ", input[i]);
    }

    getTime(&s);
    bubbleSortCPUReference(verificationInput, length, sortDescending);
    if(memcmp(input, verificationInput, length*sizeof(cl_uint)) == 0) {
        printf("\nPassed!\n");
    } else {
        printf("\nFailed\n");
    }
    getTime(&t);
    printTime(&s, &t, "Check: ", "\n");

    getTime(&s);
	/* Release memory */
    free(verificationInput);
    verificationInput = NULL;
    error_code = clFree(command_queue, input);
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
    getTime(&t);
    printTime(&s, &t, "Free: ", "\n");

    return 0;
}
