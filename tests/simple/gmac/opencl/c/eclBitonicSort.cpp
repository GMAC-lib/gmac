#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <gmac/opencl.h>

#include "utils.h"
#include "debug.h"

#include "../eclBitonicSortKernel.cl"

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
	gmactime_t s, t, S, T;

	cl_uint seed = 123;
	cl_uint sortDescending = 0;
	cl_uint *input = NULL;
	cl_uint *verificationInput;
	cl_uint length = 1024;
	assert(eclCompileSource(code) == eclSuccess);

	getTime(&s);
	// Alloc
	assert(eclMalloc((void **)&input, length * sizeof(cl_uint)) == eclSuccess);
	verificationInput = (cl_uint *) malloc(length*sizeof(cl_uint));
	if(verificationInput == NULL)
		return 0;
	getTime(&t);
	printTime(&s, &t, "Alloc: ", "\n");

	getTime(&S);
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
	ecl_kernel kernel;
	assert(eclGetKernel("bitonicSort", &kernel) == eclSuccess);
	assert(eclSetKernelArgPtr(kernel, 0, input) == eclSuccess);
	assert(eclSetKernelArg(kernel, 3, sizeof(length), &length) == eclSuccess);
	assert(eclSetKernelArg(kernel, 4, sizeof(sortDescending), &sortDescending) == eclSuccess);
	for(cl_uint stage = 0; stage < numStages; ++stage) {
		/* stage of the algorithm */
		assert(eclSetKernelArg(kernel, 1, sizeof(stage), &stage) == eclSuccess);
		/* Every stage has stage+1 passes. */
		for(cl_uint passOfStage = 0; passOfStage < stage + 1; ++passOfStage) {
			/* pass of the current stage */
			assert(eclSetKernelArg(kernel, 2, sizeof(passOfStage), &passOfStage) == eclSuccess);
			assert(eclCallNDRange(kernel, 1, NULL, globalThreads, localThreads) == eclSuccess);
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
	getTime(&T);
	printTime(&S, &T, "Total: ", "\n");

	getTime(&s);
	free(verificationInput);
	verificationInput = NULL;
	eclReleaseKernel(kernel);

	eclFree(input);
	getTime(&t);
	printTime(&s, &t, "Free: ", "\n");
	return 0;
}
