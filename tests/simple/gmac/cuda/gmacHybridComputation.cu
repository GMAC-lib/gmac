#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <sys/time.h>
#include <omp.h>
#include <semaphore.h>

#include <pthread.h>

#include <gmac.h>
#define CUDA_ERRORS
#include "debug.h"
#include "utils.h"

#define MAXDOUBLE DBL_MAX
#define MINDOUBLE DBL_MIN

#define MAX(a, b) ((a) > (b)) ? (a) : (b)
#define MIN(a, b) ((a) < (b)) ? (a) : (b)
#define BANDWIDTH(s, t) ((s) * sizeof(float) * 8.0 / 1000.0 / (t))

#define THREADS 2
#define THREAD_TIMING 0
#define THREAD_HOST 0
#define THREAD_GMAC 1


const char *pageLockedStr = "GMAC_PAGE_LOCKED";
const bool pageLockedDefault = false;
bool pageLocked = false;

typedef unsigned long long usec_t;
typedef struct {
	double in, max_in, min_in;
	double out, max_out, min_out;
} stamp_t;

const size_t buff_elems = 16 * 1024;
const size_t block_elems = 128;

const int cpu_job = 1024;
const int gpu_job = 64 * 1024;

inline usec_t get_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	usec_t tm = tv.tv_usec + 1000000 * tv.tv_sec;
	return tm;
}

__global__ void gpuB(uint8_t *p, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i > buff_elems) return;
    for(int j = 0; j < n; j++) {
        p[i]++;
    }
}

__global__ void gpuD(uint8_t *p, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i > buff_elems) return;
    for(int j = 0; j < n; j++) {
        p[i]++;
    }
}

void cpuA(uint8_t *p, int n)
{
    for(int i = 0; i < buff_elems; i++) {
        for(int j = 0; j < n; j++) {
            p[i]++;
        }
    }
}

void cpuC(uint8_t *p, int n)
{
    for(int i = 0; i < buff_elems; i++) {
        for(int j = 0; j < n; j++) {
            p[i]++;
        }
    }
}

void cpuE(uint8_t *p, int n)
{
    for(int i = 0; i < buff_elems; i++) {
        for(int j = 0; j < n; j++) {
            p[i]++;
        }
    }
}

const int REPS = 100;

int main(int argc, char *argv[])
{
    setParam<bool>(&pageLocked, pageLockedStr, pageLockedDefault);

    uint8_t * dev;
    uint8_t * host1, * host2, * host3;

    dim3 Db(block_elems);
	dim3 Dg(buff_elems / block_elems);
	if(buff_elems % block_elems) Db.x++;

    struct timeval s, t;

    sem_t sem;
    sem_init(&sem, 0, 0);
    if((host1 = (uint8_t *) malloc(buff_elems * sizeof(float))) == NULL)
        FATAL();
    if((host2 = (uint8_t *) malloc(buff_elems * sizeof(float))) == NULL)
        FATAL();
    if((host3 = (uint8_t *) malloc(buff_elems * sizeof(float))) == NULL)
        FATAL();

    for(int j = 1; j <= gpu_job; j *= 2) {
        printf("%d ", j);
    }
    printf("\n");
    omp_set_num_threads(THREADS);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        // Alloc & init input data
        if(tid == THREAD_GMAC) {
            if (gmacMalloc((void **) &dev, buff_elems * sizeof(float)) != gmacSuccess)
                FATAL();
        }

        for(int i = 1; i <= cpu_job; i *= 2) {
            printf("%d ", i);
            #pragma omp barrier
            for(int j = 1; j <= gpu_job; j *= 2) {
                if(tid == THREAD_TIMING) gettimeofday(&s, NULL);
                for(int k = 0; k < REPS; k++) {
                    if (tid == THREAD_GMAC) {
                        gpuB<<<Dg, Db>>>(gmacPtr(dev), j); 
                        gmacThreadSynchronize();
                        gmacMemcpy(host2, dev, buff_elems * sizeof(float));
                        sem_post(&sem);
                        gpuD<<<Dg, Db>>>(gmacPtr(dev), j); 
                        gmacThreadSynchronize();
                        gmacMemcpy(host3, dev, buff_elems * sizeof(float));
                        sem_post(&sem);
                    } else {
                        cpuA(host1, i);
                        sem_wait(&sem);
                        cpuC(host2, i);
                        sem_wait(&sem);
                        cpuE(host3, i);
                    }
                    #pragma omp barrier
                }
                if(tid == THREAD_TIMING) {
                    gettimeofday(&t, NULL);
                    printTime(&s, &t, "", " ");
                    fflush(stdout);
                }
            }
            printf("\n");
        }

        if (tid == THREAD_GMAC) {
            gmacFree(dev);
        }
    }

    free(host1);
    free(host2);
    free(host3);
}
