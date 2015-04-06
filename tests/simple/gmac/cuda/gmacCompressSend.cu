#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include <semaphore.h>

#include <gmac/cuda.h>

#include "utils.h"
#include "debug.h"
#include "barrier.h"

#include "gmacCompress.h"

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
		a[i] = 10.0 * rand() / RAND_MAX;
	}
}

void nextStage(stage_t *current, stage_t *next)
{
	if(next != NULL) {
		gmac_sem_wait(&next->free, 1);
		next->next_in = current->out;
		next->next_out = current->in;
		gmacSendReceive(next->id);
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
	gmacError_t ret;
    gmactime_t s, t;

    barrier_wait(&barrierInit);

	dim3 Db(blockSize, blockSize);
	dim3 Dg(width / blockSize, height / blockSize);
	if(width % blockSize) Dg.x++;
	if(height % blockSize) Dg.y++;


	for(unsigned i = 0; i < frames; i++) {
        getTime(&s);
		ret = gmacMalloc((void **)&s_dct.in, width * height * sizeof(float));
		assert(ret == gmacSuccess);
		ret = gmacMalloc((void **)&s_dct.out, width * height * sizeof(float));
		assert(ret == gmacSuccess);
        getTime(&t);
        printTime(&s, &t, "DCT-Malloc: ", "\n");

        getTime(&s);
		__randInit(s_dct.in, width * height);
        getTime(&t);
        printTime(&s, &t, "DCT-Init: ", "\n");

        getTime(&s);
		dct<<<Dg, Db>>>(gmacPtr(s_dct.out), gmacPtr(s_dct.in), width, height);
		ret = gmacThreadSynchronize();
		assert(ret == gmacSuccess);
        getTime(&t);
        printTime(&s, &t, "DCT-Run: ", "\n");

        getTime(&s);
		gmac_sem_wait(&s_quant.free, 1);
		s_quant.next_in = s_dct.out;
		s_quant.next_out = s_dct.in;
		gmacSendReceive(s_quant.id);
        getTime(&t);
        printTime(&s, &t, "DCT-SendRecv: ", "\n");
	}

    getTime(&s);
	ret = gmacMalloc((void **)&s_dct.in, width * height * sizeof(float));
	assert(ret == gmacSuccess);
	ret = gmacMalloc((void **)&s_dct.out, width * height * sizeof(float));
	assert(ret == gmacSuccess);
    getTime(&t);
    printTime(&s, &t, "DCT-Malloc: ", "\n");

    getTime(&s);
	gmac_sem_wait(&s_quant.free, 1);
	s_quant.next_in = s_dct.out;
	s_quant.next_out = s_dct.in;
	gmacSendReceive(s_quant.id);
    getTime(&t);
    printTime(&s, &t, "DCT-SendRecv: ", "\n");

    getTime(&s);
	ret = gmacMalloc((void **)&s_dct.in, width * height * sizeof(float));
	assert(ret == gmacSuccess);
	ret = gmacMalloc((void **)&s_dct.out, width * height * sizeof(float));
	assert(ret == gmacSuccess);
    getTime(&t);
    printTime(&s, &t, "DCT-Malloc: ", "\n");

    getTime(&s);
	gmac_sem_wait(&s_quant.free, 1);
	s_quant.next_in = s_dct.out;
	s_quant.next_out = s_dct.in;
	gmacSendReceive(s_quant.id);
    getTime(&t);
    printTime(&s, &t, "DCT-SendRecv: ", "\n");

	return NULL;
}

void *quant_thread(void *args)
{
	gmacError_t ret;
    gmactime_t s, t;

    barrier_wait(&barrierInit);

	dim3 Db(blockSize, blockSize);
	dim3 Dg(width / blockSize, height / blockSize);
	if(width % blockSize) Dg.x++;
	if(height % blockSize) Dg.y++;

    getTime(&s);
	gmac_sem_post(&s_quant.free, 1);
	nextStage(&s_quant, &s_idct);
    getTime(&t);
    printTime(&s, &t, "Quant-SendRecv: ", "\n");

	for(unsigned i = 0; i < frames; i++) {
        getTime(&s);
		quant<<<Dg, Db>>>(gmacPtr(s_quant.out), gmacPtr(s_quant.in), width, height, 1e-6);
		ret = gmacThreadSynchronize();
		assert(ret == gmacSuccess);
        getTime(&t);
        printTime(&s, &t, "Quant-Run: ", "\n");
		
        getTime(&s);
		nextStage(&s_quant, &s_idct);
        getTime(&t);
        printTime(&s, &t, "Quant-SendRecv: ", "\n");
	}

	// Move one stage the pipeline stages the pipeline
	getTime(&s);
	nextStage(&s_quant, &s_idct);
    getTime(&t);
    printTime(&s, &t, "Quant-SendRecv: ", "\n");

	return NULL;
}

void *idct_thread(void *args)
{
	gmacError_t ret;
    gmactime_t s, t;

    barrier_wait(&barrierInit);

	dim3 Db(blockSize, blockSize);
	dim3 Dg(width / blockSize, height / blockSize);
	if(width % blockSize) Dg.x++;
	if(height % blockSize) Dg.y++;

    getTime(&s);
	gmac_sem_post(&s_idct.free, 1);
	gmacSendReceive(s_dct.id);
	nextStage(&s_idct, NULL);
    getTime(&t);
    printTime(&s, &t, "IDCT-SendRecv: ", "\n");

    getTime(&s);
	gmacSendReceive(s_dct.id);
    getTime(&t);
	nextStage(&s_idct, NULL);
    getTime(&t);
    printTime(&s, &t, "IDCT-SendRecv: ", "\n");


	for(unsigned i = 0; i < frames; i++) {
        getTime(&s);
		idct<<<Dg, Db>>>(gmacPtr(s_idct.out), gmacPtr(s_idct.in), width, height);
		ret = gmacThreadSynchronize();
		assert(ret == gmacSuccess);
        getTime(&t);
        printTime(&s, &t, "IDCT-Run: ", "\n");

        getTime(&s);
		assert(gmacFree(s_idct.in) == gmacSuccess);
		assert(gmacFree(s_idct.out) == gmacSuccess);
        getTime(&t);
        printTime(&s, &t, "IDCT-Free: ", "\n");

        getTime(&s);
		gmacSendReceive(s_dct.id);
		nextStage(&s_idct, NULL);
        getTime(&t);
        printTime(&s, &t, "IDCT-SendRecv: ", "\n");
	}

    getTime(&s);
	gmacFree(s_idct.in);
	gmacFree(s_idct.out);
    getTime(&t);
    printTime(&s, &t, "IDCT-Free: ", "\n");

	return NULL;
}


int main(int argc, char *argv[])
{
	gmactime_t s, t;
	setParam<unsigned>(&width, widthStr, widthDefault);
	setParam<unsigned>(&height, heightStr, heightDefault);
	setParam<unsigned>(&frames, framesStr, framesDefault);

	gmac_sem_init(&s_quant.free, 0);
	gmac_sem_init(&s_idct.free, 0);

	srand(time(NULL));

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
