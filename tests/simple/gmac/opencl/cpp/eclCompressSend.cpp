#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <semaphore.h>

#include <gmac/opencl>

#include "utils.h"
#include "debug.h"
#include "barrier.h"

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

typedef struct stage {
	thread_t id;
	gmac_sem_t free;
	float *in;
	float *out;
	float *next_in;
	float *next_out;
} stage_t;

stage_t s_dct, s_quant, s_idct;


void __randInit(float *a, size_t size)
{
	for(unsigned i = 0; i < size; i++) {
		a[i] = 10.0f * rand() / RAND_MAX;
	}
}

void nextStage(stage_t *current, stage_t *next)
{
	if(next != NULL) {
		gmac_sem_wait(&next->free, 1);
		next->next_in = current->out;
		next->next_out = current->in;
		ecl::deviceSendReceive(next->id);
	}
	if(current != NULL) {
		current->in = current->next_in;
		current->out = current->next_out;
		gmac_sem_post(&current->free, 1);
	}
}

barrier_t barrierInit;

void *dct_thread(void *args)
{
	gmactime_t s, t;

	barrier_wait(&barrierInit);

	ecl::config localSize(blockSize, blockSize);
	ecl::config globalSize(width, height);
	if(width  % blockSize) globalSize.x += blockSize;
	if(height % blockSize) globalSize.y += blockSize;
	ecl::error err;
	ecl::kernel k("dct", err);
	assert(err == eclSuccess);

	assert(k.setArg(2, width)  == eclSuccess);
	assert(k.setArg(3, height) == eclSuccess);


	for(unsigned i = 0; i < frames; i++) {
		getTime(&s);
		s_dct.in = new (ecl::allocator) float[width * height];
		assert(s_dct.in != NULL);
		s_dct.out = new (ecl::allocator) float[width * height];
		assert(s_dct.out != NULL);
		getTime(&t);
		printTime(&s, &t, "DCT:Malloc: ", "\n");

		getTime(&s);
		__randInit(s_dct.in, width * height);
		getTime(&t);
		printTime(&s, &t, "DCT:Init: ", "\n");

		getTime(&s);
		assert(k.setArg(0, s_dct.out)    == eclSuccess);
		assert(k.setArg(1, s_dct.in)     == eclSuccess);
		assert(k.callNDRange(globalSize, localSize) == eclSuccess);
		getTime(&t);
		printTime(&s, &t, "DCT:Run: ", "\n");

		getTime(&s);
		gmac_sem_wait(&s_quant.free, 1);
		s_quant.next_in = s_dct.out;
		s_quant.next_out = s_dct.in;
		ecl::deviceSendReceive(s_quant.id);
		getTime(&t);
		printTime(&s, &t, "DCT:SendRecv: ", "\n");
	}

	getTime(&s);
	s_dct.in = new (ecl::allocator) float[width * height];
	assert(s_dct.in != NULL);
	s_dct.out = new (ecl::allocator) float[width * height];
	assert(s_dct.out != NULL);
	getTime(&t);
	printTime(&s, &t, "DCT:Malloc: ", "\n");

	getTime(&s);
	gmac_sem_wait(&s_quant.free, 1);
	s_quant.next_in = s_dct.out;
	s_quant.next_out = s_dct.in;
	ecl::deviceSendReceive(s_quant.id);
	getTime(&t);
	printTime(&s, &t, "DCT:SendRecv: ", "\n");

	getTime(&s);
	s_dct.in = new (ecl::allocator) float[width * height];
	assert(s_dct.in != NULL);
	s_dct.out = new (ecl::allocator) float[width * height];
	assert(s_dct.out != NULL);
	getTime(&t);
	printTime(&s, &t, "DCT:Malloc: ", "\n");

	getTime(&s);
	gmac_sem_wait(&s_quant.free, 1);
	s_quant.next_in = s_dct.out;
	s_quant.next_out = s_dct.in;
	ecl::deviceSendReceive(s_quant.id);
	getTime(&t);
	printTime(&s, &t, "DCT:SendRecv: ", "\n");

	return NULL;
}

void *quant_thread(void *args)
{
	gmactime_t s, t;

	barrier_wait(&barrierInit);

	getTime(&s);
	gmac_sem_post(&s_quant.free, 1);
	nextStage(&s_quant, &s_idct);
	getTime(&t);
	printTime(&s, &t, "Quant:SendRecv: ", "\n");

	ecl::config localSize(blockSize, blockSize);
	ecl::config globalSize(width, height);
	if(width  % blockSize) globalSize.x += blockSize;
	if(height % blockSize) globalSize.y += blockSize;
	ecl::error err;
	ecl::kernel k("quant", err);
	assert(err == eclSuccess);

	assert(k.setArg(2, width)       == eclSuccess);
	assert(k.setArg(3, height)      == eclSuccess);
	assert(k.setArg(4, float(1e-6)) == eclSuccess);

	for(unsigned i = 0; i < frames; i++) {
		getTime(&s);
		assert(k.setArg(0, s_quant.in)  == eclSuccess);
		assert(k.setArg(1, s_quant.out) == eclSuccess);
		assert(k.callNDRange(globalSize, localSize) == eclSuccess);
		getTime(&t);
		printTime(&s, &t, "Quant:Run: ", "\n");

		getTime(&s);
		nextStage(&s_quant, &s_idct);
		getTime(&t);
		printTime(&s, &t, "Quant:SendRecv: ", "\n");
	}

	// Move one stage the pipeline stages the pipeline
	getTime(&s);
	nextStage(&s_quant, &s_idct);
	getTime(&t);
	printTime(&s, &t, "Quant:SendRecv: ", "\n");

	return NULL;
}

void *idct_thread(void *args)
{
	gmactime_t s, t;

	barrier_wait(&barrierInit);

	getTime(&s);
	gmac_sem_post(&s_idct.free, 1);
	ecl::deviceSendReceive(s_dct.id);
	nextStage(&s_idct, NULL);
	getTime(&t);
	printTime(&s, &t, "IDCT:SendRecv: ", "\n");

	getTime(&s);
	gmac_sem_post(&s_idct.free, 1);
	ecl::deviceSendReceive(s_dct.id);
	getTime(&t);
	nextStage(&s_idct, NULL);
	getTime(&t);
	printTime(&s, &t, "IDCT:SendRecv: ", "\n");

	ecl::config localSize(blockSize, blockSize);
	ecl::config globalSize(width, height);
	if(width  % blockSize) globalSize.x += blockSize;
	if(height % blockSize) globalSize.y += blockSize;
	ecl::error err;
	ecl::kernel k("idct", err);
	assert(err == eclSuccess);

	assert(k.setArg(2, width)   == eclSuccess);
	assert(k.setArg(3, height)  == eclSuccess);

	for(unsigned i = 0; i < frames; i++) {
		getTime(&s);
		assert(k.setArg(0, s_idct.in)  == eclSuccess);
		assert(k.setArg(1, s_idct.out) == eclSuccess);
		assert(k.callNDRange(globalSize, localSize) == eclSuccess);
		getTime(&t);
		printTime(&s, &t, "IDCT:Run: ", "\n");

		getTime(&s);
		assert(ecl::free(s_idct.in) == eclSuccess);
		assert(ecl::free(s_idct.out) == eclSuccess);
		getTime(&t);
		printTime(&s, &t, "IDCT:Free: ", "\n");

		getTime(&s);
		ecl::deviceSendReceive(s_dct.id);
		nextStage(&s_idct, NULL);
		getTime(&t);
		printTime(&s, &t, "IDCT:SendRecv: ", "\n");
	}

	getTime(&s);
	ecl::free(s_idct.in);
	ecl::free(s_idct.out);
	getTime(&t);
	printTime(&s, &t, "IDCT:Free: ", "\n");

	return NULL;
}


int main(int argc, char *argv[])
{
	gmactime_t s, t;
	setParam<unsigned>(&width, widthStr, widthDefault);
	setParam<unsigned>(&height, heightStr, heightDefault);
	setParam<unsigned>(&frames, framesStr, framesDefault);

	assert(eclCompileSource(kernel_code) == eclSuccess);

	gmac_sem_init(&s_quant.free, 0);
	gmac_sem_init(&s_idct.free, 0);

	srand(unsigned(time(NULL)));

	getTime(&s);

	barrier_init(&barrierInit, 3);

	s_dct.id = thread_create(dct_thread, NULL);
	s_quant.id = thread_create(quant_thread, NULL);
	s_idct.id = thread_create(idct_thread, NULL);

	thread_wait(s_dct.id);
	thread_wait(s_quant.id);
	thread_wait(s_idct.id);

	getTime(&t);

	printTime(&s, &t, "Total: ", "\n");

	return 0;
}
