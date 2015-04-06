#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cassert>

#include <gmac/opencl>

#include "utils.h"

#include "eclBinarySearchKernel.cl"

const char *findMeStr = "GMAC_FINDME";
const cl_uint findMeDefault = 200;
cl_uint findMe = findMeDefault;
const char *globalLowerBoundStr = "GMAC_GLABALLOWERBOUND";
const cl_uint globalLowerBoundDefault = 0;
cl_uint globalLowerBound = globalLowerBoundDefault;
const char *globalUpperBoundStr = "GMAC_GLOBALUPPERBOUND";
const cl_uint globalUpperBoundDefault = 0;
cl_uint globalUpperBound = globalUpperBoundDefault;
const char *subdivSizeStr = "GMAC_SUBDIVSIZE";
const cl_uint subdivSizeDefault = 0;
cl_uint subdivSize = subdivSizeDefault;

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
    unsigned vecSize = 512;
    unsigned int numSubdivisions = 256;
	ecl::error ret;

    ret = ecl::compileSource(code);
	assert(ret == eclSuccess);
    setParam<cl_uint>(&findMe, findMeStr, findMeDefault);
    setParam<cl_uint>(&globalLowerBound, globalLowerBoundStr, globalLowerBoundDefault);
    setParam<cl_uint>(&globalUpperBound, globalUpperBoundStr, globalUpperBoundDefault);
    setParam<cl_uint>(&subdivSize, subdivSizeStr, subdivSizeDefault);

    // Alloc data
    input = new (ecl::allocator) cl_uint[vecSize * sizeof(cl_uint)];
    output = new (ecl::allocator) cl_uint[sizeof(cl_uint4)];
    assert(input != NULL);
    assert(output != NULL);

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

    globalLowerBound = 0;
    globalUpperBound = vecSize - 1;
    subdivSize = (globalUpperBound - globalLowerBound + 1)/numSubdivisions;
    cl_uint isElementFound = 0;

    if((input[0] > findMe) || (input[vecSize-1] < findMe)) {
        output[0] = 0;
        output[1] = vecSize - 1;
        output[2] = 0;
        printf("Not find %d\n", findMe);

        return 0;
    }

    output[3] = 1;

    ecl::config globalSize(globalThreads[0]);
    ecl::config localSize(localThreads[0]);  
	ecl::config globalWorkOffset(0);  
    ecl::kernel kernel("binarySearch", ret);
    assert(ret == eclSuccess);
#ifndef __GXX_EXPERIMENTAL_CXX0X__
    ret = kernel.setArg(0, output);
	assert(ret == eclSuccess);
    ret = kernel.setArg(1, input);
	assert(ret == eclSuccess);
    ret = kernel.setArg(2, findMe);
	assert(ret == eclSuccess);
    while(subdivSize > 1 && output[3] != 0) {
        output[3] = 0;
        ret = kernel.setArg(3, globalLowerBound);
		assert(ret == eclSuccess);
        ret = kernel.setArg(4, globalUpperBound);
		assert(ret == eclSuccess);
        ret = kernel.setArg(5, subdivSize);
		assert(ret == eclSuccess);
        ret = kernel.callNDRange(globalSize,localSize, globalWorkOffset);
		assert(ret == eclSuccess);
        globalLowerBound = output[0];
        globalUpperBound = output[1];
        subdivSize = (globalUpperBound - globalLowerBound + 1)/numSubdivisions;
    }
#else
    while(subdivSize > 1 && output[3] != 0) {
        output[3] = 0;
        ret = kernel(globalSize, localSize)(input, output, findMe, globalLowerBound, globalUpperBound, subdivSize);
		assert(ret == eclSuccess);
        globalLowerBound = output[0];
        globalUpperBound = output[1];
        subdivSize = (globalUpperBound - globalLowerBound + 1)/numSubdivisions;
    }
#endif

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

    ecl::free(input);
    ecl::free(output);

    printf("all over %f\n", 0.0);
    return 0;
}
