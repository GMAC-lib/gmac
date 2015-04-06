#include <stdio.h>
#include <cstring>
#include <gmac/cuda.h>

#include "barrier.h"
#include "utils.h"

#define THREADS 16
#define ITERATIONS 50

const unsigned size = 1024 * 1024;
const unsigned totalSize = THREADS * size;

unsigned *ptr;
barrier_t barrier;

__global__
void inc(unsigned *a, unsigned size)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= size) return;

	a[i] += 1;
}

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
    assert(gmacMalloc((void **)&ptr, totalSize * sizeof(unsigned)) == gmacSuccess);

    // Call the kernel
    unsigned globalSize = totalSize;

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

    for (unsigned i = 0; i < ITERATIONS; i++) {
        inc<<<dim3(totalSize / 512), dim3(512)>>>(gmacPtr(ptr), globalSize);
        assert(gmacThreadSynchronize() == gmacSuccess);
        sentinel++;
        //printf("Iteration %u\n", i);
        barrier_wait(&barrier);
        barrier_wait(&barrier);
    }

    for (unsigned i = 0; i < THREADS; i++) {
         thread_wait(threads[i]);
    }

    //printf("Sentinel %u\n", sentinel);
    for (unsigned i = 0; i < totalSize; i++) {
        if (ptr[i] != 2 * ITERATIONS) {
            printf("Pos: %u (%u). %u vs %u\n", i, i % size, ptr[i], 2 * ITERATIONS);
            abort();
        }
    }

	gmacFree(ptr);

    return 0;
}
