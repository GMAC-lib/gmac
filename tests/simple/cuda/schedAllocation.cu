#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <sys/time.h>

#include <pthread.h>

#include <cassert>

#include <cuda.h>
#define CUDA_ERRORS
#include "debug.h"
#include "utils.h"
#include "barrier.h"

#define MAXDOUBLE DBL_MAX
#define MINDOUBLE DBL_MIN

#define MAX(a, b) ((a) > (b)) ? (a) : (b)
#define MIN(a, b) ((a) < (b)) ? (a) : (b)
#define BANDWIDTH(s, t) ((s) * sizeof(float) * 8.0 / 1000.0 / (t))

__global__ void inc(float *c, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= size) return;

    c[i]++;
}

const char *nIterStr = "GMAC_NITER";
const uint32_t nIterDefault = 1;
uint32_t nIter;

const size_t buffElems = 16 * 1024 * 1024;
const size_t blockElems = 512;

struct param_t {
    uint8_t id;
    size_t size;
};

barrier_t barrier;
pthread_mutex_t lock;

void * thread(void * _param)
{
    float * dev;

    param_t * param = (param_t *) _param;

#if 0
    cudaSetDevice(param->id);
#endif

    barrier_wait(&barrier);
    pthread_mutex_lock(&lock);

    //cudaSetDevice(param->id % 2);

    if(cudaMalloc((void **)&dev, buffElems * sizeof(float)) != cudaSuccess) {
        printf("Thread: %d failed in attempt 1\n", param->id);
        if(cudaMalloc((void **)&dev, buffElems * sizeof(float)) != cudaSuccess) {
            printf("Thread: %d failed in attempt 2\n", param->id);
            pthread_mutex_unlock(&lock);
            barrier_wait(&barrier);
            return NULL;
        } else {
            printf("Thread: %d succeeded in attempt 2\n", param->id);
        }
    }

    pthread_mutex_unlock(&lock);

    int id;
    cudaGetDevice(&id);
    printf("Thread: %d -> %d\n", param->id, id);

    dim3 Db(blockElems);
    dim3 Dg(buffElems / blockElems);
    if(buffElems % blockElems) Dg.x++;

    inc<<<Dg, Db>>>(dev, buffElems);
    
    barrier_wait(&barrier);

	// Release memory
	cudaFree(dev);
    return NULL;
}


int main(int argc, char *argv[])
{
    setParam<uint32_t>(&nIter, nIterStr, nIterDefault);

    pthread_t threads[nIter];
	param_t params[nIter];

    barrier_init(&barrier, nIter);
    pthread_mutex_init(&lock, NULL);

    for (uint32_t i = 0; i < nIter; i++) {
        params[i].id = i;
        pthread_create(&threads[i], NULL, thread, &params[i]);
    }

    for (uint32_t i = 0; i < nIter; i++) {
        pthread_join(threads[i], NULL);
    }

    barrier_destroy(&barrier);
    pthread_mutex_destroy(&lock);
}
