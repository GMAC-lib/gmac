/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication and is exactly the same as
 * Chapter 7 of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * CUBLAS provides high-performance matrix multiplication.
 */

// includes, system
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>

// includes, project
#include <gmac/cuda.h>
#include <gmac/vm.h>

// includes, utils and debug
#include "debug.h"
#include "utils.h"

// includes, kernels
#include "vmMatrixMulKernel.cu"

const char * WAStr = "GMAC_WA";
const char * HAStr = "GMAC_HA";
const char * WBStr = "GMAC_WB";
const char * HBStr = "GMAC_HB";
const char * checkStr = "GMAC_CHECK";

const size_t WADefault = (32 * BLOCK_SIZE); // Matrix A width
const size_t HADefault = (32 * BLOCK_SIZE); // Matrix A height
const size_t WBDefault = (32 * BLOCK_SIZE); // Matrix B width
const size_t HBDefault = (32 * BLOCK_SIZE); // Matrix B height
const bool checkDefault = 0; // Matrix B height

static size_t WA = 0; // Matrix A width
static size_t HA = 0; // Matrix A height
static size_t WB = 0; // Matrix B width
static size_t HB = 0; // Matrix B height
static bool check = checkDefault; // Matrix B height

#define WC WB  // Matrix C width 
#define HC HA  // Matrix C height

static float * A, * B, * C;

size_t elemsC;
size_t sizeC;

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void
computeGold(float* C, const float* A, const float* B, size_t hA, size_t wA, size_t wB)
{
    for (size_t i = 0; i < hA; ++i) {
        for (size_t j = 0; j < wB; ++j) {
            double sum = 0;
            for (size_t k = 0; k < wA; ++k) {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (float)sum;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
    setParam<size_t>(&WA, WAStr, WADefault);
    setParam<size_t>(&HA, HAStr, HADefault);
    setParam<size_t>(&WB, WBStr, WBDefault);
    setParam<size_t>(&HB, HBStr, HBDefault);
    setParam<bool>(&check, checkStr, checkDefault);

    if (HB != WA) {
        fprintf(stderr, "Error: WA and HB must be equal\n");
        abort();
    }

    gmactime_t s, t;

    size_t elemsA = WA * HA;
    size_t elemsB = WB * HB;
    elemsC = WC * HC;
    size_t sizeA = sizeof(float) * elemsA;
    size_t sizeB = sizeof(float) * elemsB;
    sizeC = sizeof(float) * elemsC;

    printf("Elems: %zd\n", elemsA);
    printf("Elems: %zd\n", elemsB);
    printf("Elems: %zd\n", elemsC);


    // allocate memory for matrices A and B
    getTime(&s);
    if (gmacMalloc((void**) &A, sizeA) != gmacSuccess) {
        fprintf(stderr, "Error allocating A");
        abort();
    }
    if (gmacMalloc((void**) &B, sizeB) != gmacSuccess) {
        fprintf(stderr, "Error allocating B");
        abort();
    }
    if (gmacMalloc((void**) &C, sizeC) != gmacSuccess) {
        fprintf(stderr, "Error allocating C");
        abort();
    }

    // initialize matricesmatrices
    randInitMax(A, 100.f, elemsA);
    randInitMax(B, 100.f, elemsB);

    // Alloc output data
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");


    // Call the kernel
    getTime(&s);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(unsigned(WC / threads.x), unsigned(HC / threads.y));
    matrixMul<<< grid, threads >>>(gmacPtr(C), gmacPtr(A), gmacPtr(B), WA, WB, 0);
    if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
    getTime(&t);
    printTime(&s, &t, "Run: ", "\n");

    if (check == 1) {
        // compute reference solution
        getTime(&s);

        // check result
        float err = 0.0;
        printf("Computing host matrix mul. Please wait...\n");
        float* reference = (float *) malloc(sizeC);
        computeGold(reference, A, B, HA, WA, WB);

        for (unsigned i = 0; i < elemsC; i++) {
            err += fabsf(reference[i] - C[i]);
        }
        getTime(&t);
        printTime(&s, &t, "Check: ", "\n");

        fprintf(stderr, "Error: %f\n", err);
        // clean up memory
        free(reference);
    }

    gmacFree(A);
    gmacFree(B);
    gmacFree(C);

    return 0;
}
