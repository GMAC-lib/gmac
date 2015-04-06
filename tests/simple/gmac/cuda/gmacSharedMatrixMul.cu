#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>

#include <gmac/cuda.h>

#include "debug.h"
#include "utils.h"

#include "gmacMatrixMulKernel.cu"

const char * nIterStr = "GMAC_NITER";
const char * WAStr = "GMAC_WA";
const char * HAStr = "GMAC_HA";
const char * WBStr = "GMAC_WB";
const char * HBStr = "GMAC_HB";

const int nIterDefault = 4;
const int WADefault = (40 * BLOCK_SIZE); // Matrix A width
const int HADefault = (40 * BLOCK_SIZE); // Matrix A height
const int WBDefault = (40 * BLOCK_SIZE); // Matrix B width
const int HBDefault = (40 * BLOCK_SIZE); // Matrix B height

static int nIter = 0;
static int WA = 0; // Matrix A width
static int HA = 0; // Matrix A height
static int WB = 0; // Matrix B width
static int HB = 0; // Matrix B height

#define WC WB  // Matrix C width 
#define HC HA  // Matrix C height

static float * A, * B;
struct param {
	int i;
	float * ptr;
    const char *name;
};

unsigned elemsC;
unsigned sizeC;

void
computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < wA; ++k) {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (float)sum;
        }
}


void *
matrixMulThread(void * ptr)
{
    static char buffer[1024];
	struct param *p = (struct param *) ptr;

    // timers
    gmactime_t s, t;

    assert(gmacMalloc((void**) &p->ptr, sizeC) == gmacSuccess);

    // Call the kernel
	getTime(&s);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(WC / threads.x, (HC / nIter) / threads.y);
    matrixMul<<< grid, threads >>>(gmacPtr(p->ptr), gmacPtr(A), gmacPtr(B), WA, WB, p->i * elemsC);
	if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
	getTime(&t);
    snprintf(buffer, 1024, "%s-Run: ", p->name);
	printTime(&s, &t, buffer, "\n");

    return NULL;
}

float doTest(float * A, float * B, unsigned elemsA, unsigned elemsB, unsigned elemsC, const char *name)
{
    static char buffer[1024];
    thread_t * threads = new thread_t[nIter];
	param * params = new param[nIter];

    gmactime_t s, t;

	getTime(&s);
    // initialize matrices
    valueInit(A, 100.f, elemsA);
    valueInit(B, 100.f, elemsB);
	getTime(&t);
    snprintf(buffer, 1024, "%s-Init: ", name);
	printTime(&s, &t, buffer, "\n");

    for (int n = 0; n < nIter; n++) {
		params[n].i = n;
        params[n].name = name;
		threads[n] = thread_create(matrixMulThread, &(params[n]));
	}

	for (int n = 0; n < nIter; n++) {
		thread_wait(threads[n]);
	}

    // compute reference solution
	getTime(&s);
    float err = 0;
    float* reference = (float *) malloc(sizeC * nIter);
    computeGold(reference, A, B, HA, WA, WB);
    for (int n = 0; n < nIter; n++) {
        err += checkError(reference + n * elemsC, params[n].ptr, elemsC);
    }
    getTime(&t);
    snprintf(buffer, 1024, "%s-Check: ", name);
	printTime(&s, &t, buffer, "\n");

    // clean up memory
    free(reference);
    getTime(&s);
    for (int n = 0; n < nIter; n++) {
        assert(gmacFree(params[n].ptr) == gmacSuccess);
    }
    getTime(&t);
    snprintf(buffer, 1024, "%s-Free: ", name);
	printTime(&s, &t, buffer, "\n");

    delete [] params;
    delete [] threads;

    return err;
}


int
main(int argc, char** argv)
{
    gmactime_t s, t;

	setParam<int>(&nIter, nIterStr, nIterDefault);
	setParam<int>(&WA, WAStr, WADefault);
	setParam<int>(&HA, HAStr, HADefault);
	setParam<int>(&WB, WBStr, WBDefault);
	setParam<int>(&HB, HBStr, HBDefault);

    assert(nIter != 0);

    assert((HA/BLOCK_SIZE) % nIter == 0);
    assert(HB == WA);

    unsigned elemsA = WA * HA;
    unsigned elemsB = WB * HB;
             elemsC = WC * HC / nIter;
    unsigned sizeA = sizeof(float) * elemsA;
    unsigned sizeB = sizeof(float) * elemsB;
             sizeC = sizeof(float) * elemsC;

    // allocate memory for matrices A and B
	getTime(&s);
    assert(gmacGlobalMalloc((void**) &A, sizeA, GMAC_GLOBAL_MALLOC_REPLICATED) == gmacSuccess);
    assert(gmacGlobalMalloc((void**) &B, sizeB, GMAC_GLOBAL_MALLOC_REPLICATED) == gmacSuccess);
    getTime(&t);
    printTime(&s, &t, "Replicated:Alloc: ", "\n");
    float err_replicated = doTest(A, B, elemsA, elemsB, elemsC, "Replicated");
    getTime(&s);
    assert(gmacFree(A) == gmacSuccess);
	assert(gmacFree(B) == gmacSuccess);
    getTime(&t);
    printTime(&s, &t, "Replicated:Free: ", "\n");


    // allocate memory for matrices A and B
	getTime(&s);
    assert(gmacGlobalMalloc((void**) &A, sizeA, GMAC_GLOBAL_MALLOC_CENTRALIZED) == gmacSuccess);
    assert(gmacGlobalMalloc((void**) &B, sizeB, GMAC_GLOBAL_MALLOC_CENTRALIZED) == gmacSuccess);
    getTime(&t);
    printTime(&s, &t, "Centralized:Alloc: ", "\n");
    float err_centralized = doTest(A, B, elemsA, elemsB, elemsC, "Centralized");
    getTime(&s);
    assert(gmacFree(A) == gmacSuccess);
	assert(gmacFree(B) == gmacSuccess);
    getTime(&t);
    printTime(&s, &t, "Centralized:Free: ", "\n");

    return fabsf(err_replicated) == 0.0f && fabsf(err_centralized) == 0.0f;
}
