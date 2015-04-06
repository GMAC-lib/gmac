#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include "gmac/opencl"

#include "utils.h"
#include "debug.h"

#include "../eclBitonicSortKernel.cl"

#define GROUP_SIZE 1

const char *sortDescendingStr = "GMAC_SORTDESCENDING";
const cl_uint sortDescendingDefault = 0;
cl_uint sortDescending = sortDescendingDefault;
const char *lengthStr = "GMAC_LENGGTH";
const cl_uint lengthDefault = 1024;
cl_uint length = lengthDefault;
const char *stageStr = "GMAC_STAGE";
const cl_uint stageDefault = 0;
cl_uint stage = stageDefault;
const char *passOfStageStr = "GMAC_PASSOFSTAGE";
const cl_uint passOfStageDefault = 0;
cl_uint passOfStage = passOfStageDefault;

void swapIfFirstIsGreater(cl_uint *a, cl_uint *b)
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
	gmactime_t s, t, S , T;

	cl_uint seed = 123;
	cl_uint *input = NULL;
	cl_uint *verificationInput;

	assert(ecl::compileSource(code) == eclSuccess);
	setParam<cl_uint>(&sortDescending, sortDescendingStr, sortDescendingDefault);
	setParam<cl_uint>(&length, lengthStr, lengthDefault);
	setParam<cl_uint>(&stage, stageStr, stageDefault);
	setParam<cl_uint>(&passOfStage, passOfStageStr, passOfStageDefault);

	getTime(&s);
	// Alloc
	input = new (ecl::allocator) cl_uint[length * sizeof(cl_uint)];
	assert(input != NULL);
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
	for(cl_uint temp = length; temp > 1; temp >>= 1)
		++numStages;
	ecl::config globalSize(length / 2);
	ecl::config localSize(GROUP_SIZE);
	ecl::error err;
	ecl::kernel kernel("bitonicSort", err);
	assert(err == eclSuccess);
#ifndef __GXX_EXPERIMENTAL_CXX0X__
	assert(kernel.setArg(0, input) == eclSuccess);
	assert(kernel.setArg(3, length) == eclSuccess);
	assert(kernel.setArg(4, sortDescending) == eclSuccess);
	for(stage = 0; stage < numStages; ++stage) {
		/* stage of the algorithm */
		assert(kernel.setArg(1, stage) == eclSuccess);
		/* Every stage has stage+1 passes. */
		for(passOfStage = 0; passOfStage < stage + 1; ++passOfStage) {
			/* pass of the current stage */
			assert(kernel.setArg(2, passOfStage) == eclSuccess);
			assert(kernel.callNDRange(globalSize, localSize) == eclSuccess);
		}
	}
#else
	for(cl_uint stage = 0; stage < numStages; ++stage) {
		for(cl_uint passOfStage = 0; passOfStage < stage + 1; ++passOfStage) {
			assert(kernel(globalSize, localSize)(input, stage, passOfStage, length, sortDescending) == eclSuccess);
		}
	}
#endif
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
	ecl::free(input);
	getTime(&t);
	printTime(&s, &t, "Free: ", "\n");
	return 0;
}