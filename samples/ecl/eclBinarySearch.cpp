#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <gmac/opencl.h>

#include "eclBinarySearchKernel.cl"

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
	cl_uint *input, *output;
    cl_uint findMe = 200;
    unsigned vecSize = 512;
    unsigned int numSubdivisions = 256;
	ecl_error ret;

    ret = eclCompileSource(code);
	assert(ret == eclSuccess);

    // Alloc data
    ret = eclMalloc((void **)&input, vecSize * sizeof(cl_uint));
	assert(ret == eclSuccess);
    ret = eclMalloc((void **)&output, sizeof(cl_uint4));
	assert(ret == eclSuccess);

    // random initialisation of input
    cl_uint max = vecSize * 20;
    input[0] = 0;
    for(cl_uint i = 1; i < vecSize; i++)
        input[i] = input[i-1] + (cl_uint) (max * rand()/(float)RAND_MAX);

    // Print the input data
    printf("Sorted data: ");
    for(cl_uint i = 0; i < vecSize; i++)
        printf("%d ", input[i]);

    // Call the kernel
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
    ecl_kernel kernel;
    ret = eclGetKernel("binarySearch", &kernel);
	assert(ret == eclSuccess);
    ret = eclSetKernelArgPtr(kernel, 0, output);
	assert(ret == eclSuccess);
    ret = eclSetKernelArgPtr(kernel, 1, input);
	assert(ret == eclSuccess);
    ret = eclSetKernelArg(kernel, 2, sizeof(findMe), &findMe);
	assert(ret == eclSuccess);
    while(subdivSize > 1 && output[3] != 0) {
        output[3] = 0;
        ret = eclSetKernelArg(kernel, 3, sizeof(globalLowerBound), &globalLowerBound);
		assert(ret == eclSuccess);
        ret = eclSetKernelArg(kernel, 4, sizeof(globalUpperBound), &globalUpperBound);
		assert(ret == eclSuccess);
        ret = eclSetKernelArg(kernel, 5, sizeof(subdivSize), &subdivSize);
		assert(ret == eclSuccess);
        ret = eclCallNDRange(kernel, 1, NULL, globalThreads, localThreads);
		assert(ret == eclSuccess);
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

    printf("\nl = %d, u = %d, isfound = %d, fm = %d\n", globalLowerBound, globalUpperBound, isElementFound, findMe);

    cl_int verified = binarySearchCPUReference(input, output, findMe, vecSize);
    /* compare the results and see if they match */
    if(verified) {
        printf("Passed!\n");
    } else {
        printf("Failed\n");
    }

    eclReleaseKernel(kernel);

    eclFree(input);
    eclFree(output);
    printf("all over %f\n", 0.0);
    return 0;
}
