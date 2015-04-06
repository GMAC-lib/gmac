#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <gmac/opencl>

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
	ecl::config localSize(blockSize);
	ecl::config globalSize(vecSize/blockSize);
	gmactime_t s, t;
	if(vecSize % blockSize) globalSize.x++;
	globalSize.x *= localSize.x;

	// Alloc & init input data
	getTime(&s);
	if(ecl::malloc((void **)&a, vecSize * sizeof(float)) != eclSuccess)
		CUFATAL();
	if(ecl::malloc((void **)&b, vecSize * sizeof(float)) != eclSuccess)
		CUFATAL();
	if(ecl::malloc((void **)&c, vecSize * sizeof(float)) != eclSuccess)
		CUFATAL();
	getTime(&t);
	timeAlloc += getTimeStamp(t) - getTimeStamp(s);

	getTime(&s);
	ecl::memset(a, 0, vecSize * sizeof(float));
	ecl::memset(b, 0, vecSize * sizeof(float));
	getTime(&t);
	timeMemset += getTimeStamp(t) - getTimeStamp(s);

	barrier_wait(&ioBefore);

	ecl::error err;
	ecl::kernel kernelSet("vecSet", err);
	assert(err == eclSuccess);

	err = kernelSet.setArg(0, a);
	assert(err == eclSuccess);
	err = kernelSet.setArg(1, vecSize);
	assert(err == eclSuccess);

	ecl::kernel kernelMove("vecMove", err);
	assert(err == eclSuccess);

	err = kernelMove.setArg(0, c);
	assert(err == eclSuccess);
	err = kernelMove.setArg(1, a);
	assert(err == eclSuccess);
	err = kernelMove.setArg(2, vecSize);
	assert(err == eclSuccess);

	for (int i = 0; i < ITERATIONS; i++) {
		getTime(&s);
		float val = float(i);
		err = kernelSet.setArg(2, val);
		assert(err == eclSuccess);
		err = kernelSet.callNDRange(globalSize, localSize);
		assert(err == eclSuccess);

		getTime(&t);
		timeRun += getTimeStamp(t) - getTimeStamp(s);
		barrier_wait(&ioAfter);
		getTime(&s);
		err = kernelMove.callNDRange(globalSize, localSize);
		assert(err == eclSuccess);
		getTime(&t);
		timeRun += getTimeStamp(t) - getTimeStamp(s);
		barrier_wait(&ioBefore);
	}

	barrier_wait(&ioBefore);

	ecl::kernel kernelAccum("vecAccum", err);
	assert(err == eclSuccess);

	err = kernelAccum.setArg(0, c);
	assert(err == eclSuccess);
	err = kernelAccum.setArg(1, a);
	assert(err == eclSuccess);
	err = kernelAccum.setArg(2, vecSize);
	assert(err == eclSuccess);

	for (int i = ITERATIONS - 1; i >= 0; i--) {
		barrier_wait(&ioBefore);
		barrier_wait(&ioAfter);
		getTime(&s);
		err = kernelAccum.callNDRange(globalSize, localSize);
		assert(err == eclSuccess);
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

	ecl::free(a);
	ecl::free(b);
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

	assert(ecl::compileSource(kernel) == eclSuccess);

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
