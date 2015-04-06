#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac/cuda.h>

#include "utils.h"
#include "semaphore.h"

const char *nIterStr = "GMAC_NITER";
const char *vecSizeStr = "GMAC_VECSIZE";
const char *roundsStr = "GMAC_ROUNDS";

const int nIterDefault = 4;
const size_t vecSizeDefault = 1024 * 1024;
const unsigned roundsDefault = 4;

int nIter = 0;
size_t vecSize = 0;
unsigned rounds = 0;
const size_t blockSize = 512;

static thread_t *nThread;
static int *ids;
static float **a;
static gmac_sem_t init;

__global__ void inc(float *a, float f, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= size) return;

    a[i] += f;
}

void *chain(void *ptr)
{
    int *id = (int *)ptr;
    gmacError_t ret = gmacSuccess;
    int n = 0, m = 0;

    ret = gmacMalloc((void **)&a[*id], vecSize * sizeof(float));
    assert(ret == gmacSuccess);
    valueInit(a[*id], *id, vecSize);
    int next = (*id == nIter - 1) ? 0 : *id + 1;
    dim3 Db(blockSize);
    dim3 Dg(vecSize / blockSize);
    if(vecSize % blockSize) Dg.x++;

    gmac_sem_wait(&init, 1);

    for(unsigned i = 0; i < rounds; i++) {
        int current = *id - i;
        if(current < 0) current += nIter;
        // Call the kernel
        inc<<<Dg, Db>>>(gmacPtr(a[current]), *id, vecSize);
        if(gmacThreadSynchronize() != gmacSuccess) abort();

        // Pass the context
        n++;
        gmacSendReceive(nThread[next]);
        m++;
    }
    int current = *id - rounds;
    if(current < 0) current += nIter;

    fprintf(stderr,"%d (Thread %d): %d sends\t%d receives\n", current, *id, n, m);
    float error = 0;
    for(unsigned i = 0; i < vecSize; i++) {
        error += (a[current][i]);
    }
    fprintf(stderr,"%d (Thread %d): Error %f\n", current, *id, error / 1024);

    assert(int(error) % 1024 == 0);

	gmacFree(a[current]);

	return NULL;
}


int main(int argc, char *argv[])
{
	int n = 0;

	setParam<int>(&nIter, nIterStr, nIterDefault);
	setParam<size_t>(&vecSize, vecSizeStr, vecSizeDefault);
	setParam<unsigned>(&rounds, roundsStr, roundsDefault);
    gmac_sem_init(&init, 0);

	nThread = (thread_t *)malloc(nIter * sizeof(thread_t));
	ids = (int *)malloc(nIter * sizeof(int));
	a = (float **)malloc(nIter * sizeof(float **));

	for(n = 0; n < nIter; n++) {
		ids[n] = n;
		nThread[n] = thread_create(chain, &ids[n]);
	}

    fprintf(stderr,"Ready... Steady\n");
	for(n = 0; n < nIter; n++) gmac_sem_post(&init, 1);
    fprintf(stderr,"Go!\n");
	
	for(n = 0; n < nIter; n++) {
		thread_wait(nThread[n]);
	}
    fprintf(stderr,"Done!\n");

	free(ids);
	free(nThread);

    return 0;
}
