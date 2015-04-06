#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>

#include <gmac/opencl>

#include "debug.h"
#include "utils.h"

#include "../eclMatrixMulKernel.cl"

#define BLOCK_SIZE 8

const char * WAStr = "GMAC_WA";
const char * HAStr = "GMAC_HA";
const char * WBStr = "GMAC_WB";
const char * HBStr = "GMAC_HB";
const char * checkStr = "GMAC_CHECK";

const unsigned WADefault = (32 * BLOCK_SIZE); // Matrix A width
const unsigned HADefault = (32 * BLOCK_SIZE); // Matrix A height
const unsigned WBDefault = (32 * BLOCK_SIZE); // Matrix B width
const unsigned HBDefault = (32 * BLOCK_SIZE); // Matrix B height
const int checkDefault = true; // Check results

static unsigned WA = 0; // Matrix A width
static unsigned HA = 0; // Matrix A height
static unsigned WB = 0; // Matrix B width
static unsigned HB = 0; // Matrix B height
static bool check = checkDefault; // Check results

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

int
main(int argc, char** argv)
{
	ecl::error ret;
	ret = ecl::compileSource(code);
	assert(ret == eclSuccess);

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
	//	getTime(&s);

	A = new (ecl::allocator) float[sizeA];
	assert(A != NULL);
	B = new (ecl::allocator) float[sizeB];
	assert(B != NULL);
	C = new (ecl::allocator) float[sizeC];
	assert(C != NULL);

	getTime(&t);
	printTime(&s, &t, "Alloc: ", "\n");

	getTime(&s);
	valueInit(A, 100.f, elemsA);
	valueInit(B, 100.f, elemsB);
	getTime(&t);
	printTime(&s, &t, "Init: ", "\n");
	getTime(&s);

	ecl::config globalSize(WC,HC);
	ecl::config localSize(BLOCK_SIZE, BLOCK_SIZE);
	ecl::config globalWorkOffset(0);

	ecl::kernel kernel("matrixMulSimple",ret);
	assert(ret == eclSuccess);
#ifndef __GXX_EXPERIMENTAL_CXX0X__
	ret = kernel.setArg(0, C);
	assert(ret == eclSuccess);
	ret = kernel.setArg(1, A);
	assert(ret == eclSuccess);
	ret = kernel.setArg(2, B);
	assert(ret == eclSuccess);

	int param = int(WA);
	ret = kernel.setArg(3, param);
	assert(ret == eclSuccess);
	param = int(WB);                                 
	ret = kernel.setArg(4, param);
	assert(ret == eclSuccess);
	ret = kernel.callNDRange(globalSize, localSize, globalWorkOffset);
	assert(ret == eclSuccess);
#else
	assert(kernel(C, A, B, WA, WB)(globalSize, localSize, globalWorkOffset) == eclSuccess);
#endif
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
	operator delete(A, ecl::allocator);
	operator delete(B, ecl::allocator);
	operator delete(C, ecl::allocator);

	getTime(&t);
	printTime(&s, &t, "Free: ", "\n");

	fprintf(stderr, "Error: %f\n", err);

	return fabsf(err) != 0.0f;
}
