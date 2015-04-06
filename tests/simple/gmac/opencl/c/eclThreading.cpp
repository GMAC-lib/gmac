#include <stdio.h>
#include <cstring>
#include <gmac/opencl.h>

#include "barrier.h"
#include "utils.h"

#define THREADS 16
#define ITERATIONS 50

const unsigned size = 64 * 1024;
const unsigned totalSize = THREADS * size;

unsigned *ptr;
barrier_t barrier;

const char *kernel = "\
__kernel void inc(__global unsigned *a, unsigned size)\
{\
    unsigned i = get_global_id(0);\
    if(i >= size) return;\
\
	a[i] += 1;\
}\
";

unsigned sentinel = 0;

void *func(void *p)
{
    unsigned id = *(unsigned *) p;
    unsigned offset = id * size;

    barrier_wait(&barrier);

    for (unsigned i = 0; i < ITERATIONS; i++) {
        barrier_wait(&barrier);
        for (unsigned j = 0; j < size; j++) {
            ptr[offset + j]++;
        }
        if (id == 0) sentinel++;
        barrier_wait(&barrier);
    }

    return NULL;
}

int main(int argc, char *argv[])
{
    assert(eclCompileSource(kernel) == eclSuccess);

    assert(eclMalloc((void **)&ptr, totalSize * sizeof(unsigned)) == eclSuccess);

    // Call the kernel
    size_t globalSize = size_t(totalSize);

    thread_t threads[THREADS];
    unsigned ids[THREADS];

    barrier_init(&barrier, THREADS + 1);

    for (unsigned i = 0; i < THREADS; i++) {
        ids[i] = i;
        threads[i] = thread_create(func, &ids[i]);
    }

    for (unsigned i = 0; i < totalSize; i++) {
        ptr[i] = 0;
    }

    barrier_wait(&barrier);

    ecl_kernel kernel;

    assert(eclGetKernel("inc", &kernel) == eclSuccess);

    assert(eclSetKernelArgPtr(kernel, 0, ptr) == eclSuccess);
    assert(eclSetKernelArg(kernel, 1, sizeof(totalSize), &totalSize) == eclSuccess);

    for (unsigned i = 0; i < ITERATIONS; i++) {
        assert(eclCallNDRange(kernel, 1, NULL, &globalSize, NULL) == eclSuccess);
        sentinel++;
        printf("Iteration %u\n", i);
        barrier_wait(&barrier);
        barrier_wait(&barrier);
    }

    for (unsigned i = 0; i < THREADS; i++) {
         thread_wait(threads[i]);
    }

    printf("Sentinel %u\n", sentinel);
    for (unsigned i = 0; i < totalSize; i++) {
        if (ptr[i] != 2 * ITERATIONS) {
            fprintf(stderr, "Pos: %u (%u). %u vs %u\n", i, i % size, ptr[i], 2 * ITERATIONS);
            abort();
        }
    }

    eclReleaseKernel(kernel);

    eclFree(ptr);

    return 0;
}
