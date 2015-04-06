#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <semaphore.h>

#include <gmac/opencl>

#include "utils.h"
#include "debug.h"

#include "../eclCompressCommon.cl"

const char *widthStr = "GMAC_WIDTH";
const char *heightStr = "GMAC_HEIGHT";
const char *framesStr = "GMAC_FRAMES";

const unsigned widthDefault = 128;
const unsigned heightDefault = 128;
const unsigned framesDefault = 32;

unsigned width = 0;
unsigned height = 0;
unsigned frames = 0;
const unsigned blockSize = 16;

static float *quant_in, *idct_in;

static thread_t dct_id, quant_id, idct_id;
static gmac_sem_t quant_data, idct_data;
static gmac_sem_t quant_free, idct_free;

double timeDCTAlloc = 0.0;
double timeDCTInit  = 0.0;
double timeDCTCopy  = 0.0;
double timeDCTRun   = 0.0;
double timeDCTFree  = 0.0;

double timeQuantAlloc = 0.0;
double timeQuantCopy  = 0.0;
double timeQuantRun   = 0.0;
double timeQuantFree  = 0.0;

double timeIDCTAlloc = 0.0;
double timeIDCTRun   = 0.0;
double timeIDCTFree  = 0.0;

void __randInit(float *a, unsigned size)
{
	for(unsigned i = 0; i < size; i++) {
		a[i] = 10.0f * rand() / RAND_MAX;
	}
}

void *dct_thread(void *args)
{
	float *in, *out;
	gmactime_t s, t;

	getTime(&s);
	in = new (ecl::allocator) float[width * height];
	assert(in != NULL);
	out = new (ecl::allocator) float[width * height];
	assert(out != NULL);
	getTime(&t);
	timeDCTAlloc += getTimeStamp(t) - getTimeStamp(s);

	ecl::config localSize(blockSize, blockSize);
	ecl::config globalSize(width, height);
	if(width  % blockSize) globalSize.x += blockSize;
	if(height % blockSize) globalSize.y += blockSize;
	ecl::error err;
	ecl::kernel k("dct", err);
	assert(err == eclSuccess);

#ifndef __GXX_EXPERIMENTAL_CXX0X__
	assert(k.setArg(0, out)    == eclSuccess);
	assert(k.setArg(1, in)     == eclSuccess);
	assert(k.setArg(2, width)  == eclSuccess);
	assert(k.setArg(3, height) == eclSuccess);

	for(unsigned i = 0; i < frames; i++) {
		getTime(&s);
		__randInit(in, width * height);
		getTime(&t);
		timeDCTInit += getTimeStamp(t) - getTimeStamp(s);

		getTime(&s);
		assert(k.callNDRange(globalSize, localSize) == eclSuccess);
		getTime(&t);
		timeDCTRun += getTimeStamp(t) - getTimeStamp(s);

		getTime(&s);
		gmac_sem_wait(&quant_free, 1); /* Wait for quant to use its data */
		ecl::memcpy(quant_in, out, width * height * sizeof(float));
		gmac_sem_post(&quant_data, 1); /* Notify to Quant that data is ready */
		getTime(&t);
		timeDCTCopy += getTimeStamp(t) - getTimeStamp(s);
	}
#else
	for(unsigned i = 0; i < frames; i++) {
		getTime(&s);
		__randInit(in, width * height);
		getTime(&t);
		timeDCTInit += getTimeStamp(t) - getTimeStamp(s);

		getTime(&s);
		assert(k(out, in, width, height)(globalSize, localSize) == eclSuccess);
		getTime(&t);
		timeDCTRun += getTimeStamp(t) - getTimeStamp(s);

		getTime(&s);
		gmac_sem_wait(&quant_free, 1); /* Wait for quant to use its data */
		ecl::memcpy(quant_in, out, width * height * sizeof(float));
		gmac_sem_post(&quant_data, 1); /* Notify to Quant that data is ready */
		getTime(&t);
		timeDCTCopy += getTimeStamp(t) - getTimeStamp(s);
	}
#endif
	getTime(&s);
	operator delete(in, ecl::allocator);
	operator delete(out, ecl::allocator);
	getTime(&t);
	timeDCTFree += getTimeStamp(t) - getTimeStamp(s);

	return NULL;
}

void *quant_thread(void *args)
{
	float *out;
	gmactime_t s, t;

	getTime(&s);
	quant_in = new (ecl::allocator) float[width * height];
	assert(quant_in != NULL);
	out = new (ecl::allocator) float[width * height];
	assert(out != NULL);
	getTime(&t);
	timeQuantAlloc += getTimeStamp(t) - getTimeStamp(s);

	ecl::config localSize(blockSize, blockSize);
	ecl::config globalSize(width, height);
	if(width  % blockSize) globalSize.x += blockSize;
	if(height % blockSize) globalSize.y += blockSize;
	ecl::error err;
	ecl::kernel k("quant", err);
	assert(err == eclSuccess);

#ifndef __GXX_EXPERIMENTAL_CXX0X__
	assert(k.setArg(0, quant_in)    == eclSuccess);
	assert(k.setArg(1, out)         == eclSuccess);
	assert(k.setArg(2, width)       == eclSuccess);
	assert(k.setArg(3, height)      == eclSuccess);
	assert(k.setArg(4, float(1e-6)) == eclSuccess);

	gmac_sem_post(&quant_free, 1);

	for(unsigned i = 0; i < frames; i++) {
		getTime(&s);
		gmac_sem_wait(&quant_data, 1);	/* Wait for data to be processed */
		assert(k.callNDRange(globalSize, localSize) == eclSuccess);
		getTime(&t);
		timeQuantRun += getTimeStamp(t) - getTimeStamp(s);

		getTime(&s);
		gmac_sem_wait(&idct_free, 1); /* Wait for IDCT to use its data */
		ecl::memcpy(idct_in, out, width * height * sizeof(float));
		gmac_sem_post(&quant_free, 1); /* Notify to DCT that Quant is waiting for data */
		gmac_sem_post(&idct_data, 1); /* Nodify to IDCT that data is ready */
		getTime(&t);
		timeQuantCopy += getTimeStamp(t) - getTimeStamp(s);
	}
#else
	gmac_sem_post(&quant_free, 1);

	for(unsigned i = 0; i < frames; i++) {
		getTime(&s);
		gmac_sem_wait(&quant_data, 1);	/* Wait for data to be processed */
		assert(k(quant_in, out, width, height, float(1e-6))(globalSize, localSize) == eclSuccess);
		getTime(&t);
		timeQuantRun += getTimeStamp(t) - getTimeStamp(s);

		getTime(&s);
		gmac_sem_wait(&idct_free, 1); /* Wait for IDCT to use its data */
		ecl::memcpy(idct_in, out, width * height * sizeof(float));
		gmac_sem_post(&quant_free, 1); /* Notify to DCT that Quant is waiting for data */
		gmac_sem_post(&idct_data, 1); /* Nodify to IDCT that data is ready */
		getTime(&t);
		timeQuantCopy += getTimeStamp(t) - getTimeStamp(s);
	}
#endif
	getTime(&s);
	operator delete(quant_in, ecl::allocator);
	operator delete(out, ecl::allocator);
	getTime(&t);
	timeQuantFree += getTimeStamp(t) - getTimeStamp(s);

	return NULL;
}

void *idct_thread(void *args)
{
	float *out;
	gmactime_t s, t;

	getTime(&s);
	idct_in = new (ecl::allocator) float[width * height];
	assert(idct_in != NULL);
	out = new (ecl::allocator) float[width * height];
	assert(out != NULL);
	getTime(&t);
	timeIDCTAlloc += getTimeStamp(t) - getTimeStamp(s);

	ecl::config localSize(blockSize, blockSize);
	ecl::config globalSize(width, height);
	if(width  % blockSize) globalSize.x += blockSize;
	if(height % blockSize) globalSize.y += blockSize;
	ecl::error err;
	ecl::kernel k("idct", err);
	assert(err == eclSuccess);
#ifndef __GXX_EXPERIMENTAL_CXX0X__
	assert(k.setArg(0, idct_in) == eclSuccess);
	assert(k.setArg(1, out)     == eclSuccess);
	assert(k.setArg(2, width)   == eclSuccess);
	assert(k.setArg(3, height)  == eclSuccess);

	gmac_sem_post(&idct_free, 1);

	for(unsigned i = 0; i < frames; i++) {
		getTime(&s);
		gmac_sem_wait(&idct_data, 1);
		assert(k.callNDRange(globalSize, localSize) == eclSuccess);
		gmac_sem_post(&idct_free, 1);
		getTime(&t);
		timeIDCTRun += getTimeStamp(t) - getTimeStamp(s);
	}
#else
	gmac_sem_post(&idct_free, 1);

	for(unsigned i = 0; i < frames; i++) {
		getTime(&s);
		gmac_sem_wait(&idct_data, 1);
		assert(k(idct_in, out, width, height)(globalSize, localSize) == eclSuccess);
		gmac_sem_post(&idct_free, 1);
		getTime(&t);
		timeIDCTRun += getTimeStamp(t) - getTimeStamp(s);
	}
#endif
	memset(out, 0, width * height * sizeof(float));
	getTime(&s);
	operator delete(idct_in, ecl::allocator);
	operator delete(out, ecl::allocator);
	getTime(&t);
	timeIDCTFree += getTimeStamp(t) - getTimeStamp(s);

	return NULL;
}


int main(int argc, char *argv[])
{
	gmactime_t s,t;
	setParam<unsigned>(&width, widthStr, widthDefault);
	setParam<unsigned>(&height, heightStr, heightDefault);
	setParam<unsigned>(&frames, framesStr, framesDefault);

	assert(eclCompileSource(kernel_code) == eclSuccess);

	gmac_sem_init(&quant_data, 0); 
	gmac_sem_init(&quant_free, 0); 
	gmac_sem_init(&idct_data,  0); 
	gmac_sem_init(&idct_free,  0); 

	srand(unsigned(time(NULL)));

	getTime(&s);

	dct_id = thread_create(dct_thread, NULL);
	quant_id = thread_create(quant_thread, NULL);
	idct_id = thread_create(idct_thread, NULL);

	thread_wait(dct_id);
	thread_wait(quant_id);
	thread_wait(idct_id);

	getTime(&t);

	fprintf(stdout, "DCT-Alloc: %f\n", timeDCTAlloc / 1e6);
	fprintf(stdout, "DCT-Init: %f\n", timeDCTInit / 1e6);
	fprintf(stdout, "DCT-Run: %f\n", timeDCTRun / 1e6);
	fprintf(stdout, "DCT-Copy: %f\n", timeDCTCopy / 1e6);
	fprintf(stdout, "DCT-Free: %f\n", timeDCTFree / 1e6);

	fprintf(stdout, "Quant-Alloc: %f\n", timeQuantAlloc / 1e6);
	fprintf(stdout, "Quant-Run: %f\n", timeQuantRun / 1e6);
	fprintf(stdout, "Quant-Copy: %f\n", timeQuantCopy / 1e6);
	fprintf(stdout, "Quant-Free: %f\n", timeQuantFree / 1e6);

	fprintf(stdout, "IDCT-Alloc: %f\n", timeIDCTAlloc / 1e6);
	fprintf(stdout, "IDCT-Run: %f\n", timeIDCTRun / 1e6);
	fprintf(stdout, "IDCT-Free: %f\n", timeIDCTFree / 1e6);

	printTime(&s, &t, "Total: ", "\n");

	return 0;
}
