#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <sys/time.h>

#include <pthread.h>

#include <cuda.h>
#define CUDA_ERRORS
#include "debug.h"
#include "utils.h"

#define MAXDOUBLE DBL_MAX
#define MINDOUBLE DBL_MIN

#define MAX(a, b) ((a) > (b)) ? (a) : (b)
#define MIN(a, b) ((a) < (b)) ? (a) : (b)
#define BANDWIDTH(s, t) ((s) * sizeof(float) * 8.0 / 1000.0 / (t))

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

    // Alloc & init input data
    if(cudaMalloc((void **) &dev, buff_elems * sizeof(float)) != cudaSuccess)
        FATAL();
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
    for(int i = 1; i <= cpu_job; i *= 2) {
        printf("%d ", i);
        for(int j = 1; j <= gpu_job; j *= 2) {
            gettimeofday(&s, NULL);
            for(int k = 0; k < REPS; k++) {
                gpuB<<<Dg, Db>>>(dev, j); 
                cpuA(host1, i);
                cudaThreadSynchronize();
                cudaMemcpy(host2, dev, buff_elems * sizeof(float), cudaMemcpyDeviceToHost);
                gpuD<<<Dg, Db>>>(dev, j);
                cpuC(host2, i);
                cudaThreadSynchronize();
                cudaMemcpy(host3, dev, buff_elems * sizeof(float), cudaMemcpyDeviceToHost);
                cpuE(host3, i);
            }
            gettimeofday(&t, NULL);
            printTime(&s, &t, "", " ");
            fflush(stdout);
        }
        printf("\n");
	}

    cudaFree(dev);
    free(host1);
    free(host2);
    free(host3);
}
