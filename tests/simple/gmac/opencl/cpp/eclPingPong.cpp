#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <gmac/opencl>

#include "utils.h"
#include "semaphore.h"

const char *nIterStr = "GMAC_NITER";
const char *vecSizeStr = "GMAC_VECSIZE";
const char *roundsStr = "GMAC_ROUNDS";

const int nIterDefault = 4;
const unsigned vecSizeDefault = 1024 * 1024;
const unsigned roundsDefault = 4;

int nIter = 0;
unsigned vecSize = 0;
unsigned rounds = 0;
const unsigned blockSize = 512;

static thread_t *nThread;
static int *ids;
static float **a;
static gmac_sem_t init;

const char *kernel_code = "\
						  __kernel void inc(__global float *a, float f, unsigned size) \n \
						  {                                  \n                         \
						  unsigned i = get_global_id(0); \n                         \
						  if(i >= size) return;          \n                         \
						  \n                         \
						  a[i] += f;                     \n                         \
						  }";

void *chain(void *ptr)
{
	int *id = (int *)ptr;
	ecl::error ret = eclSuccess;
	int n = 0, m = 0;
	ret = ecl::malloc((void **)&a[*id], vecSize * sizeof(float));
	assert(ret == eclSuccess);
	valueInit(a[*id], float(*id), vecSize);

	int next = (*id == nIter - 1) ? 0 : *id + 1;

	gmac_sem_wait(&init, 1);

	ecl::kernel kernel("inc", ret);
	assert(ret == eclSuccess);
	ecl::config globalWorkOffset(0); 
	ecl::config localSize(1); 
	ecl::config globalSize(vecSize);

	for(unsigned i = 0; i < rounds; i++) {
		int current = *id - i;
		if(current < 0) current += nIter;
		// Call the kernel
		ret = kernel.setArg(0, a[current]);
		assert(ret == eclSuccess);
		ret = kernel.setArg(1, *id);
		assert(ret == eclSuccess);
		ret = kernel.setArg(2, vecSize);
		assert(ret == eclSuccess);

		ret = kernel.callNDRange(globalSize,localSize, globalWorkOffset);
		assert(ret == eclSuccess);

		// Pass the context
		n++;
		ecl::deviceSendReceive(nThread[next]);
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

	ecl::free(a[current]);

	return NULL;
}


int main(int argc, char *argv[])
{
	int n = 0;

	assert(ecl::compileSource(kernel_code) == eclSuccess);

	setParam<int>(&nIter, nIterStr, nIterDefault);
	setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);
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

