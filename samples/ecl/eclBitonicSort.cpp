#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <gmac/opencl.h>

#include "eclBitonicSortKernel.cl"

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
    cl_uint seed = 123;
    cl_uint sortDescending = 0;
    cl_uint *input = NULL;
    cl_uint *verificationInput;
    cl_uint length = 1024;
	ecl_error ret;

    ret = eclCompileSource(code);
	assert(ret == eclSuccess);

    // Alloc
    ret = eclMalloc((void **)&input, length * sizeof(cl_uint));
	assert(ret == eclSuccess);
    verificationInput = (cl_uint *) malloc(length*sizeof(cl_uint));
    if(verificationInput == NULL)
        return 0;

    /* random initialisation of input */
    srand(seed);
    const cl_uint rangeMin = 0;
    const cl_uint rangeMax = 255;
    double range = double(rangeMax - rangeMin) + 1.0;
    for(cl_uint i = 0; i < length; i++) {
        input[i] = rangeMin + (cl_uint)(range * rand() / (RAND_MAX + 1.0));
    }

    memcpy(verificationInput, input, length*sizeof(cl_uint));

    // Print the input data
    printf("Unsorted Input: ");
    for(cl_uint i = 0; i < length; i++)
        printf("%d ", input[i]);

    cl_uint numStages = 0;
    size_t globalThreads[1] = {length / 2};
    size_t localThreads[1] = {GROUP_SIZE};

    for(cl_uint temp = length; temp > 1; temp >>= 1)
        ++numStages;
    ecl_kernel kernel;
    ret = eclGetKernel("bitonicSort", &kernel);
	assert(ret == eclSuccess);
    ret = eclSetKernelArgPtr(kernel, 0, input);
	assert(ret == eclSuccess);
    ret = eclSetKernelArg(kernel, 3, sizeof(length), &length);
	assert(ret == eclSuccess);
    ret = eclSetKernelArg(kernel, 4, sizeof(sortDescending), &sortDescending);
	assert(ret == eclSuccess);
    for(cl_uint stage = 0; stage < numStages; ++stage) {
        /* stage of the algorithm */
        ret = eclSetKernelArg(kernel, 1, sizeof(stage), &stage);
		assert(ret == eclSuccess);
        /* Every stage has stage+1 passes. */
        for(cl_uint passOfStage = 0; passOfStage < stage + 1; ++passOfStage) {
            /* pass of the current stage */
            ret = eclSetKernelArg(kernel, 2, sizeof(passOfStage), &passOfStage);
			assert(ret == eclSuccess);
            ret = eclCallNDRange(kernel, 1, NULL, globalThreads, localThreads);
			assert(ret == eclSuccess);
        }
    }

    printf("Output: ");
    for(cl_uint i = 0; i < length; i++) {
        printf("%d ", input[i]);
    }

    bubbleSortCPUReference(verificationInput, length, sortDescending);
    if(memcmp(input, verificationInput, length*sizeof(cl_uint)) == 0) {
        printf("\nPassed!\n");
    } else {
        printf("\nFailed\n");
    }

    free(verificationInput);
    verificationInput = NULL;
    eclReleaseKernel(kernel);

    eclFree(input);
    return 0;
}
