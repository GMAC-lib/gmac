#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <gmac/opencl.h>

#include "utils.h"
#include "barrier.h"
#include "debug.h"

#ifdef _MSC_VER
#define VECTORA "inputset\\vectorA"
#define VECTORB "inputset\\vectorB"
#define VECTORC "inputset\\vectorC"
#else
#define VECTORA "inputset/vectorA"
#define VECTORB "inputset/vectorB"
#define VECTORC "inputset/vectorC"
#endif

const unsigned vecSize = 1024 * 1024;
const unsigned blockSize = 256;

const char *msg = "Done!";

const char *kernel = "\
					 __kernel void vecSet(__global float *_a, unsigned size, float val)\
					 {\
					 unsigned i = get_global_id(0);\
					 if(i >= size) return;\
					 \
					 _a[i] = val;\
					 }\
					 \
					 __kernel void vecAccum(__global float *_b, __global const float *_a, unsigned size)\
					 {\
					 unsigned i = get_global_id(0);\
					 if(i >= size) return;\
					 \
					 _b[i] += _a[i];\
					 }\
					 \
					 __kernel void vecMove(__global float *_a, __global const float *_b, unsigned size)\
					 {\
					 unsigned i = get_global_id(0);\
					 if(i >= size) return;\
					 \
					 _a[i] = _b[i];\
					 }\
					 ";

#define ITERATIONS 250

barrier_t ioAfter;
barrier_t ioBefore;

static void
writeFile(float *v, unsigned nmemb, int it);
static void
readFile(float *v, unsigned nmemb, int it);

float *a, *b, *c;

float error_compute, error_io;

double timeAlloc  = 0.0;
double timeMemset = 0.0;
double timeRun    = 0.0;
double timeCheck  = 0.0;
double timeFree   = 0.0;
double timeWrite  = 0.0;
double timeRead   = 0.0;

void *doTest(void *)
{
	size_t localSize = blockSize;
	size_t globalSize = vecSize / blockSize;
	gmactime_t s, t;
	if(vecSize % blockSize) globalSize++;
	globalSize *= localSize;

	// Alloc & init input data
	getTime(&s);
	if(eclMalloc((void **)&a, vecSize * sizeof(float)) != eclSuccess)
		CUFATAL();
	if(eclMalloc((void **)&b, vecSize * sizeof(float)) != eclSuccess)
		CUFATAL();
	if(eclMalloc((void **)&c, vecSize * sizeof(float)) != eclSuccess)
		CUFATAL();
	getTime(&t);
	timeAlloc += getTimeStamp(t) - getTimeStamp(s);

	getTime(&s);
	eclMemset(a, 0, vecSize * sizeof(float));
	eclMemset(b, 0, vecSize * sizeof(float));
	getTime(&t);
	timeMemset += getTimeStamp(t) - getTimeStamp(s);

	barrier_wait(&ioBefore);

	ecl_kernel kernelSet;

	assert(eclGetKernel("vecSet", &kernelSet) == eclSuccess);

	assert(eclSetKernelArgPtr(kernelSet, 0, a) == eclSuccess);
	assert(eclSetKernelArg(kernelSet, 1, sizeof(vecSize), &vecSize) == eclSuccess);

	ecl_kernel kernelMove;

	assert(eclGetKernel("vecMove", &kernelMove) == eclSuccess);

	assert(eclSetKernelArgPtr(kernelMove, 0, c) == eclSuccess);
	assert(eclSetKernelArgPtr(kernelMove, 1, a) == eclSuccess);
	assert(eclSetKernelArg(kernelMove, 2, sizeof(vecSize), &vecSize) == eclSuccess);

	for (int i = 0; i < ITERATIONS; i++) {
		getTime(&s);
		float val = float(i);
		assert(eclSetKernelArg(kernelSet, 2, sizeof(val), &val) == eclSuccess);
		assert(eclCallNDRange(kernelSet, 1, NULL, &globalSize, &localSize) == eclSuccess);

		getTime(&t);
		timeRun += getTimeStamp(t) - getTimeStamp(s);
		barrier_wait(&ioAfter);
		getTime(&s);
		assert(eclCallNDRange(kernelMove, 1, NULL, &globalSize, &localSize) == eclSuccess);
		getTime(&t);
		timeRun += getTimeStamp(t) - getTimeStamp(s);
		barrier_wait(&ioBefore);
	}

	barrier_wait(&ioBefore);

	ecl_kernel kernelAccum;
	assert(eclGetKernel("vecAccum", &kernelAccum) == eclSuccess);

	assert(eclSetKernelArgPtr(kernelAccum, 0, c) == eclSuccess);
	assert(eclSetKernelArgPtr(kernelAccum, 1, a) == eclSuccess);
	assert(eclSetKernelArg(kernelAccum, 2, sizeof(vecSize), &vecSize) == eclSuccess);

	for (int i = ITERATIONS - 1; i >= 0; i--) {
		barrier_wait(&ioBefore);
		barrier_wait(&ioAfter);
		getTime(&s);
		assert(eclCallNDRange(kernelAccum, 1, NULL, &globalSize, &localSize) == eclSuccess);
		getTime(&t);
		timeRun += getTimeStamp(t) - getTimeStamp(s);
	}


	error_compute = 0.f;
	getTime(&s);
	for(unsigned i = 0; i < vecSize; i++) {
		error_compute += b[i] - (ITERATIONS - 1)*(ITERATIONS / 2);
	}
	getTime(&t);
	timeCheck += getTimeStamp(t) - getTimeStamp(s);
	getTime(&s);
	eclReleaseKernel(kernelSet);
	eclReleaseKernel(kernelMove);
	eclReleaseKernel(kernelAccum);

	eclFree(a);
	eclFree(b);
	getTime(&t);
	timeFree += getTimeStamp(t) - getTimeStamp(s);

	return &error_compute;
}

static void
setPath(char *name, size_t len, int it)
{
	static const char path_base[] = "_ecl_file_";
	memset(name, '\0', len);
	sprintf(name, "%s%d", path_base, it);
}

static void
writeFile(float *v, unsigned nmemb, int it)
{
	char path[256];
	setPath(path, 256, it);
	gmactime_t s, t;

	getTime(&s);
	FILE * f = fopen(path, "wb");
	assert(f != NULL);
	assert(fwrite(v, sizeof(float), nmemb, f) == nmemb);
	fclose(f);
	getTime(&t);
	timeWrite += getTimeStamp(t) - getTimeStamp(s);
}

static void
readFile(float *v, unsigned nmemb, int it)
{
	char path[256];
	setPath(path, 256, it);
	gmactime_t s, t;

	getTime(&s);
	FILE * f = fopen(path, "rb");
	assert(f != NULL);
	assert(fread(v, sizeof(float), nmemb, f) == nmemb);
	fclose(f);
	getTime(&t);
	timeRead += getTimeStamp(t) - getTimeStamp(s);
}

void *doTestIO(void *)
{
	error_io = 0.0f;
	barrier_wait(&ioBefore);

	for (int i = 0; i < ITERATIONS; i++) {
		barrier_wait(&ioAfter);
		barrier_wait(&ioBefore);
		writeFile(c, vecSize, i);
	}

	barrier_wait(&ioBefore);

	for (int i = ITERATIONS - 1; i >= 0; i--) {
		barrier_wait(&ioBefore);
		readFile(a, vecSize, i);
		barrier_wait(&ioAfter);
	}

	return &error_io;
}

int main(int argc, char *argv[])
{
	thread_t tid, tidIO;

	assert(eclCompileSource(kernel) == eclSuccess);

	barrier_init(&ioAfter,2);
	barrier_init(&ioBefore, 2);

	tid = thread_create(doTest, NULL);
	tidIO = thread_create(doTestIO, NULL);

	thread_wait(tid);
	thread_wait(tidIO);

	barrier_destroy(&ioAfter);
	barrier_destroy(&ioBefore);

	fprintf(stdout, "Alloc: %f\n", timeAlloc /1e6);
	fprintf(stdout, "Memset: %f\n", timeMemset /1e6);
	fprintf(stdout, "Run: %f\n", timeRun /1e6);
	fprintf(stdout, "Check: %f\n", timeCheck /1e6);
	fprintf(stdout, "Free: %f\n", timeFree /1e6);
	fprintf(stdout, "Write: %f\n", timeWrite /1e6);
	fprintf(stdout, "Read: %f\n", timeRead /1e6);

	return error_io != 0.f;
}
