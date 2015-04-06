#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac/cuda>

#include "utils.h"
#include "debug.h"

const char *vecSizeStr = "GMAC_VECSIZE";
const unsigned vecSizeDefault = 16 * 1024 * 1024;
unsigned vecSize = vecSizeDefault;

const char *operationsStr = "GMAC_OPERATIONS";
const unsigned operationsDefault = 256 * 1024;
unsigned operations = operationsDefault;

const char *msg = "Done!";

__global__ void null(float *a)
{
}

int main(int argc, char *argv[])
{
    float *a;
    gmactime_t s, t;

    setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);
    fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

    getTime(&s);
    // Alloc data
    a = new (gmac::allocator) float[vecSize];

    assert(a != NULL);

    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    // Init input data
    getTime(&s);
    gmac::memset(a, 0, vecSize * sizeof(float));

    for(unsigned i = 0; i < operations; i++) {
        double rnd = (double(rand()) / RAND_MAX);
        unsigned pos = unsigned(rnd * (vecSize - 1));
        a[pos]++;
    }

    getTime(&t);
    printTime(&s, &t, "Init: ", "\n");

    // Call the kernel
    getTime(&s);

    null<<<1, 1>>>(gmac::ptr(a));
    assert(gmac::threadSynchronize() == gmacSuccess);

    getTime(&t);
    printTime(&s, &t, "Run: ", "\n");

    getTime(&s);
    float sum = 0.f;
    for(unsigned i = 0; i < vecSize; i++) {
        sum += a[i];
    }
    getTime(&t);
    printTime(&s, &t, "Check: ", "\n");
    fprintf(stderr, "Error: %f\n", sum - float(operations));

    gmac::free(a);

    return sum != float(operations);
}
