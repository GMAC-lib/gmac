#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac/cuda>

#include "utils.h"
#include "debug.h"

const char *xSizeStr = "GMAC_XSIZE";
const unsigned xSizeDefault = 1024;
unsigned xSize = xSizeDefault;

const char *ySizeStr = "GMAC_YSIZE";
const unsigned ySizeDefault = 128;
unsigned ySize = ySizeDefault;

const char *msg = "Done!";

__global__ void null(float *a)
{
}


int main(int argc, char *argv[])
{
    float *a;
    gmactime_t s, t;

    setParam<unsigned>(&xSize, xSizeStr, xSizeDefault);
    setParam<unsigned>(&ySize, ySizeStr, ySizeDefault);

    fprintf(stdout, "Matrix: %u x %u\n", xSize , ySize);

    getTime(&s);
    // Alloc data
    a = new (gmac::allocator) float[xSize * ySize];

    assert(a != NULL);

    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    // Init input data
    getTime(&s);
    gmac::memset(a, 0, xSize * ySize * sizeof(float));

    for (unsigned j = 0; j < xSize; j++) {
        for (unsigned i = 0; i < ySize; i++) {
            a[j + i * xSize] = 1.f;
        }
    }

    getTime(&t);
    printTime(&s, &t, "Init: ", "\n");

    // Call the kernel
    getTime(&s);

    null<<<1,1>>>(gmacPtr(a));
    assert(gmac::threadSynchronize() == gmacSuccess);

    getTime(&t);
    printTime(&s, &t, "Run: ", "\n");

    getTime(&s);
    double sum = 0.f;
    for (unsigned i = 0; i < ySize; i++) {
        for (unsigned j = 0; j < xSize; j++) {
            sum += a[j + i * xSize];
        }
    }
    getTime(&t);
    printTime(&s, &t, "Check: ", "\n");
    fprintf(stderr, "Error: %f\n", sum - double(xSize * ySize));

    gmac::free(a);

    return sum != double(xSize * ySize);
}
