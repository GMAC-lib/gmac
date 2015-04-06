#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>

#include <gmac/cuda.h>

#include "debug.h"
#include "utils.h"

#include "gmacMatrixMulKernel.cu"


const char * WAStr = "GMAC_WA";
const char * HAStr = "GMAC_HA";
const char * WBStr = "GMAC_WB";
const char * HBStr = "GMAC_HB";
const char * checkStr = "GMAC_CHECK";

const unsigned WADefault = (32 * BLOCK_SIZE); // Matrix A width
const unsigned HADefault = (32 * BLOCK_SIZE); // Matrix A height
const unsigned WBDefault = (32 * BLOCK_SIZE); // Matrix B width
const unsigned HBDefault = (32 * BLOCK_SIZE); // Matrix B height
const int checkDefault = true; // Matrix B height

static unsigned WA = 0; // Matrix A width
static unsigned HA = 0; // Matrix A height
static unsigned WB = 0; // Matrix B width
static unsigned HB = 0; // Matrix B height
static bool check = checkDefault; // Matrix B height

#define WC WB  // Matrix C width 
#define HC HA  // Matrix C height

static float * A, * B, * C;
struct param {
	int i;
	float * ptr;
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
	struct param *p = (struct param *) ptr;

    // timers
    gmactime_t s, t;

    getTime(&s);
    assert(gmacMalloc((void**) &p->ptr, sizeC) == gmacSuccess); 
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    // Call the kernel
    getTime(&s);
	if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
	getTime(&t);
	printTime(&s, &t, "Run: ", "\n");

    return NULL;
}

int
main(int argc, char** argv)
{
	setParam<unsigned>(&WA, WAStr, WADefault);
	setParam<unsigned>(&HA, HAStr, HADefault);
	setParam<unsigned>(&WB, WBStr, WBDefault);
	setParam<unsigned>(&HB, HBStr, HBDefault);
	setParam<bool>(&check, checkStr, checkDefault);

    assert(HB == WA);

    gmactime_t s, t;
    unsigned elemsA = WA * HA;
    unsigned elemsB = WB * HB;
             elemsC = WC * HC;
    unsigned sizeA = sizeof(float) * elemsA;
    unsigned sizeB = sizeof(float) * elemsB;
             sizeC = sizeof(float) * elemsC;

    // allocate memory for matrices A and B
	getTime(&s);
    assert(gmacMalloc((void**) &A, sizeA) == gmacSuccess);
    assert(gmacMalloc((void**) &B, sizeB) == gmacSuccess);
    assert(gmacMalloc((void**) &C, sizeC) == gmacSuccess);
	getTime(&t);
	printTime(&s, &t, "Alloc: ", "\n");


	getTime(&s);
    valueInit(A, 100.f, elemsA);
    valueInit(B, 100.f, elemsB);
	getTime(&t);
	printTime(&s, &t, "Init: ", "\n");

	getTime(&s);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(WC / threads.x, HC / threads.y);
    matrixMulSimple<<< grid, threads >>>(gmacPtr(C), gmacPtr(A), gmacPtr(B), WA, WB);
    gmacThreadSynchronize();
	getTime(&t);
    printTime(&s, &t, "Run: ", "\n");

    // compute reference solution
    getTime(&s);
    float err = 0.0;
    float* reference = (float *) malloc(sizeC);
    computeGold(reference, A, B, HA, WA, WB);
    for (unsigned i = 0; i < elemsC; i++) {
        err += fabsf(reference[i] - C[i]);
    }
    getTime(&t);
    printTime(&s, &t, "Check: ", "\n");

    free(reference);

    getTime(&s);
	gmacFree(A);
	gmacFree(B);
	gmacFree(C);
    getTime(&t);
    printTime(&s, &t, "Free: ", "\n");

    return fabsf(err) != 0.0f;
}
