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
#include "barrier.h"

#define MAXDOUBLE DBL_MAX
#define MINDOUBLE DBL_MIN

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define BANDWIDTH(s, t) ((s) * sizeof(float) * 8.0 / 1000.0 / (t))

const char *pageLockedStr = "GMAC_PAGE_LOCKED";
const bool pageLockedDefault = false;
bool pageLocked = false;

const char *affinity1Str = "GMAC_AFFINITY1";
const int affinity1Default = -1;
int affinity1;

const char *affinity2Str = "GMAC_AFFINITY2";
const int affinity2Default = -1;
int affinity2;

typedef unsigned long long usec_t;
typedef struct {
	double in, max_in, min_in, memcpy_in, memcpy_max_in, memcpy_min_in;
	double out, max_out, min_out, memcpy_out, memcpy_max_out, memcpy_min_out;
} stamp_t;

const size_t buff_elems = 16 * 1024 * 1024;
const size_t initial_elems = 1024;
const int iters = 256;
//const size_t block_elems = 512;

static uint8_t *cpu;
static uint8_t *cpuTmp;

inline usec_t get_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	usec_t tm = tv.tv_usec + 1000000 * tv.tv_sec;
	return tm;
}

struct param_t {
    uint8_t id;
    stamp_t *stamp;
    size_t size;
};

barrier_t barrier;
barrier_t barrier_in;
barrier_t barrier_out;

void * transfer(void * _param)
{
    uint8_t * dev;

    param_t * param = (param_t *) _param;
    stamp_t * stamp = param->stamp;

    cudaSetDevice(param->id);
#ifdef LINUX
    int affinity;
    if(param->id == 0) {
        affinity = affinity1;
    } else {
        affinity = affinity2;
    }

    if(affinity != -1) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(affinity, &set);
        if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
            printf("Error setting affinity\n");
        } else {
            printf("Thread %d: affinity: %u\n", param->id, affinity);
        }
    }
#endif

	if(cudaMalloc((void **)&dev, buff_elems * sizeof(float)) != cudaSuccess)
		CUFATAL();

	usec_t start, end;
	int j;

    for(int i = initial_elems; i <= buff_elems; i *= 2) {
        barrier_wait(&barrier);

        if (param->id == 0) {
            stamp->in = 0;
            stamp->max_in = MINDOUBLE;
            stamp->min_in = MAXDOUBLE;
            stamp->memcpy_in = 0;
            stamp->memcpy_max_in = MINDOUBLE;
            stamp->memcpy_min_in = MAXDOUBLE;
        } else if (param->id == 1) {
            stamp->out = 0;
            stamp->max_out = MINDOUBLE;
            stamp->min_out = MAXDOUBLE;
            stamp->memcpy_out = 0;
            stamp->memcpy_max_out = MINDOUBLE;
            stamp->memcpy_min_out = MAXDOUBLE;
            barrier_wait(&barrier_out);
        }

        for(j = 0; j < iters; j++) {
            if (param->id == 0) {
                barrier_wait(&barrier_out);
                start = get_time();
                if(cudaMemcpy(cpu, dev, param->size * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                    CUFATAL();
                cudaThreadSynchronize();
                end = get_time();
                stamp->in += end - start;
                stamp->min_in = MIN(stamp->min_in, (end - start));
                stamp->max_in = MAX(stamp->max_in, (end - start));
                if(pageLocked) {
                    start = get_time();
                    memcpy(cpuTmp, cpu, param->size * sizeof(float));
                    end = get_time();
                    stamp->memcpy_in += end - start;
                    stamp->memcpy_min_in = MIN(stamp->memcpy_min_in, (end - start));
                    stamp->memcpy_max_in = MAX(stamp->memcpy_max_in, (end - start));
                }
                barrier_wait(&barrier_in);
            }

            if (param->id == 1) {
                barrier_wait(&barrier_in);
                if(pageLocked) {
                    start = get_time();
                    memcpy(cpu, cpuTmp, param->size * sizeof(float));
                    end = get_time();
                    stamp->memcpy_out += end - start;
                    stamp->memcpy_min_out = MIN(stamp->memcpy_min_out, (end - start));
                    stamp->memcpy_max_out = MAX(stamp->memcpy_max_out, (end - start));
                }
                start = get_time();
                if(cudaMemcpy(dev, cpu, param->size * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
                    CUFATAL();
                cudaThreadSynchronize();
                end = get_time();
                stamp->out += end - start;
                stamp->min_out = MIN(stamp->min_out, (end - start));
                stamp->max_out = MAX(stamp->max_out, (end - start));
                barrier_wait(&barrier_out);
            }
        }
        if (param->id == 0) {
            barrier_wait(&barrier_out);
            stamp->in = stamp->in / iters;
            stamp->memcpy_in = stamp->memcpy_in / iters;
        } else if (param->id == 1) {
            stamp->out = stamp->out / iters;
            stamp->memcpy_out = stamp->memcpy_out / iters;
        }

        barrier_wait(&barrier);
    }

    
	// Release memory
	cudaFree(dev);
    return NULL;
}


int main(int argc, char *argv[])
{
    setParam<bool>(&pageLocked, pageLockedStr, pageLockedDefault);
    setParam<int>(&affinity1, affinity1Str, affinity1Default);
    setParam<int>(&affinity2, affinity2Str, affinity2Default);

	// Alloc & init input data
    if(pageLocked) {
        if(cudaHostAlloc((void **) &cpu, buff_elems * sizeof(float), cudaHostAllocPortable) != cudaSuccess)
            FATAL();
        if((cpuTmp = (uint8_t *)malloc(buff_elems * sizeof(float))) == NULL)
            FATAL();
    } else {
        if((cpu = (uint8_t *)malloc(buff_elems * sizeof(float))) == NULL)
            FATAL();
        if((cpuTmp = (uint8_t *)malloc(buff_elems * sizeof(float))) == NULL)
            FATAL();
    }

    pthread_t threads[2];
	param_t params[2];
	stamp_t stamp;

    params[0].id = 0;
    params[1].id = 1;

    params[0].stamp = &stamp;
    params[1].stamp = &stamp;

    pthread_create(&threads[0], NULL, transfer, &params[0]);
    pthread_create(&threads[1], NULL, transfer, &params[1]);

    barrier_init(&barrier, 3);
    barrier_init(&barrier_in, 2);
    barrier_init(&barrier_out, 2);

	// Transfer data
	fprintf(stdout, "#Bytes\tIn Time\tOut Time\t");
	fprintf(stdout, "In Bwd\tOut Bwd\t");
	fprintf(stdout, "In Mcpy\tOut Mcpy\t");
    /*
	fprintf(stdout, "Min In Bandwidth\tMin Out Bandwidth\t");
	fprintf(stdout, "Max In Bandwidth\tMax Out Bandwidth\n");
    */
	fprintf(stdout, "\n");
	for(int i = initial_elems; i <= buff_elems; i *= 2) {
        params[0].size = i;
        params[1].size = i;
        // Begin
        barrier_wait(&barrier);
        // End
        barrier_wait(&barrier);

		if(i * sizeof(float) > 1024 * 1024) fprintf(stdout, "%dMB\t", i * sizeof(float) / 1024 / 1024);
		else if(i * sizeof(float) > 1024) fprintf(stdout, "%dKB\t", i * sizeof(float) / 1024);
		fprintf(stdout, "%f\t%f\t", BANDWIDTH(i, stamp.in + stamp.memcpy_in), BANDWIDTH(i, stamp.out + stamp.memcpy_out));
		fprintf(stdout, "%f\t%f\t", stamp.in, stamp.out);
		fprintf(stdout, "%f\t%f\t", BANDWIDTH(i, stamp.in), BANDWIDTH(i, stamp.out));
        #if 0
		fprintf(stdout, "%f\t%f\t", BANDWIDTH(i, stamp.min_in), BANDWIDTH(i, stamp.min_out));
		fprintf(stdout, "%f\t%f\t", BANDWIDTH(i, stamp.max_in), BANDWIDTH(i, stamp.max_out));
        #endif
        fprintf(stdout, "%f\t%f\t", stamp.memcpy_in, stamp.memcpy_out);
        fprintf(stdout, "%f\t%f\t", BANDWIDTH(i, stamp.memcpy_in), BANDWIDTH(i, stamp.memcpy_out));
        #if 0
        fprintf(stdout, "%f\t%f\t", BANDWIDTH(i, stamp.memcpy_min_in), BANDWIDTH(i, stamp.memcpy_min_out));
        fprintf(stdout, "%f\t%f", BANDWIDTH(i, stamp.memcpy_max_in), BANDWIDTH(i, stamp.memcpy_max_out));
        #endif
        fprintf(stdout, "\n");
	}

    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);

    barrier_destroy(&barrier);
    barrier_destroy(&barrier_in);
    barrier_destroy(&barrier_out);

    if(pageLocked) {
        cudaFreeHost(cpu);
    } else {
        free(cpu);
    }
}
