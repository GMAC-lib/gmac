#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <sys/time.h>

#include <cuda.h>
#define CUDA_ERRORS
#include "debug.h"

#define MAXDOUBLE DBL_MAX
#define MINDOUBLE DBL_MIN

#define MAX(a, b) ((a) > (b)) ? (a) : (b)
#define MIN(a, b) ((a) < (b)) ? (a) : (b)
#define BANDWIDTH(s, t) ((s) * 8.0 / 1000.0 / (t))

typedef unsigned long long usec_t;
typedef struct {
	double in, max_in, min_in;
	double out, max_out, min_out;
} stamp_t;

const size_t buff_size = 4 * 1024 * 1024;
const size_t step_size = 2 * 1024 * 1024;
const int iters = 1024;
const size_t block_size = 512;

static uint8_t *cpu, *dev;


inline usec_t get_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	usec_t tm = tv.tv_usec + 1000000 * tv.tv_sec;
	return tm;
}

__global__ void null(uint8_t *p)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i > buff_size) return;
	p[i] = 0;
}

static void kernel(int s)
{
	dim3 Db(block_size);
	dim3 Dg(s / block_size);
	if(s % block_size) Db.x++;
	null<<<Dg, Db>>>(dev);
	if(cudaThreadSynchronize() != cudaSuccess) CUFATAL();

}

void transfer(stamp_t *stamp, int s)
{
	usec_t start, end;
	int i;

	stamp->in = stamp->out = 0;
	stamp->max_in = stamp->max_out = MINDOUBLE;
	stamp->min_in = stamp->min_out = MAXDOUBLE;
	for(i = 0; i < iters; i++) {
		start = get_time();
		if(cudaMemcpy(dev, cpu, s, cudaMemcpyHostToDevice) != cudaSuccess)
			CUFATAL();
		cudaThreadSynchronize();
		end = get_time();
		stamp->in += end - start;
		stamp->min_in = MIN(stamp->min_in, (end - start));
		stamp->max_in = MAX(stamp->max_in, (end - start));
		kernel(s);

		start = get_time();
		if(cudaMemcpy(cpu, dev, s, cudaMemcpyDeviceToHost) != cudaSuccess)
			CUFATAL();
		cudaThreadSynchronize();
		end = get_time();
		stamp->out += end - start;
		stamp->min_out = MIN(stamp->min_out, (end - start));
		stamp->max_out = MAX(stamp->max_out, (end - start));
	}
	stamp->in = stamp->in / iters;
	stamp->out = stamp->out /iters;
}


int main(int argc, char *argv[])
{
	stamp_t stamp;
	int i;

	// Alloc & init input data
	if((cpu = (uint8_t *)malloc(buff_size)) == NULL)
		FATAL();
	if(cudaMalloc((void **)&dev, buff_size) != cudaSuccess)
		CUFATAL();

	// Transfer data
	fprintf(stdout, "#Bytes\tIn Time\tOut Time\t");
	fprintf(stdout, "In Bandwidth\tOut Bandwidth\t");
	fprintf(stdout, "Min In Bandwidth\tMin Out Bandwidth\t");
	fprintf(stdout, "Max In Bandwidth\tMax Out Bandwidth\n");
	for(i = step_size; i < buff_size; i += i) {
		transfer(&stamp, i);
		if(i > 1024 * 1024) fprintf(stdout, "%.0fMB\t", 1.0 * i / 1024 / 1024);
		else if(i > 1024) fprintf(stdout, "%.0fKB\t", 1.0 * i / 1024);
		fprintf(stdout, "%f\t%f\t", stamp.in, stamp.out);
		fprintf(stdout, "%f\t%f\t", BANDWIDTH(i, stamp.in), BANDWIDTH(i, stamp.out));
		fprintf(stdout, "%f\t%f\t", BANDWIDTH(i, stamp.min_in), BANDWIDTH(i, stamp.min_out));
		fprintf(stdout, "%f\t%f\n", BANDWIDTH(i, stamp.max_in), BANDWIDTH(i, stamp.max_out));
	}

	// Release memory
	cudaFree(dev);
	free(cpu);
}
