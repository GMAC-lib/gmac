#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>

#include <gmac/opencl>

#include "debug.h"
#include "utils.h"

#include "eclMatrixMulKernel.cl"

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

struct param {
	int i;
	float * ptr;
};

typedef ecl::multi_array<float, 2> matrix;
typedef boost::multi_array<float, 2> h_matrix;

unsigned elemsC;
void
computeGold(h_matrix &C, const matrix &A, const matrix &B)
{
    assert(A.shape()[0] == B.shape()[1]);

    for (unsigned i = 0; i < A.shape()[0]; ++i) {
        for (unsigned j = 0; j < B.shape()[1]; ++j) {
            double sum = 0;
            for (unsigned k = 0; k < A.shape()[1]; ++k) {
                double a = double(A[i][k]);
                double b = double(B[k][j]);
                sum += a * b;
            }
            C[i][j] = float(sum);
        }
    }
}

int
main(int argc, char** argv)
{
    assert(eclCompileSource(code) == eclSuccess);
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

    // allocate memory for matrices A and B
	getTime(&s);
    matrix::extent_gen extents;
    matrix A(extents[HA][WA]);
    matrix B(extents[HB][WB]);
    matrix C(extents[HC][WC]);

    assert(A.origin() != NULL);
    assert(B.origin() != NULL);
    assert(C.origin() != NULL);
	getTime(&t);
	printTime(&s, &t, "Alloc: ", "\n");

	getTime(&s);
    valueInit(A.data(), 100.f, elemsA);
    valueInit(B.data(), 100.f, elemsB);
	getTime(&t);
	printTime(&s, &t, "Init: ", "\n");

	getTime(&s);
    size_t localSize[2] = { BLOCK_SIZE, BLOCK_SIZE };
    size_t globalSize[2];
    globalSize[0] = WC;
    globalSize[1] = HC;

    ecl_kernel kernel;

    assert(eclGetKernel("matrixMulSimple", &kernel) == eclSuccess);
    assert(eclSetKernelArgPtr(kernel, 0, C.data()) == eclSuccess);
    assert(eclSetKernelArgPtr(kernel, 1, A.data()) == eclSuccess);
    assert(eclSetKernelArgPtr(kernel, 2, B.data()) == eclSuccess);
    unsigned param = unsigned(A.shape()[1]);
    assert(eclSetKernelArg(kernel, 3, sizeof(int), &param) == eclSuccess);
    param          = unsigned(B.shape()[1]);                                 
    assert(eclSetKernelArg(kernel, 4, sizeof(int), &param) == eclSuccess);

    assert(eclCallNDRange(kernel, 2, NULL, globalSize, localSize) == eclSuccess);

    getTime(&t);
    printTime(&s, &t, "Run: ", "\n");

    // compute reference solution
    getTime(&s);
    float err = 0.0;
    h_matrix reference(extents[HC][WC]);
    computeGold(reference, A, B);
    for (unsigned i = 0; i < A.shape()[0]; ++i) {
        for (unsigned j = 0; j < B.shape()[1]; ++j) {
            err += fabsf(reference[i][j] - C[i][j]);
        }
    }
    getTime(&t);
    printTime(&s, &t, "Check: ", "\n");

    eclReleaseKernel(kernel);

    fprintf(stderr, "Error: %f\n", err);

    return fabsf(err) != 0.0f;
}
