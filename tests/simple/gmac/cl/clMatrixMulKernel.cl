
static const char * code = " \
#define BLOCK_SIZE 8 \n \
__kernel void \n \
matrixMulSimple( __global float* C, __global float* A, __global float* B, int wA, int wB) \n \
{ \n \
    int bx = get_group_id(0); \n \
    int by = get_group_id(1); \n \
    int tx = get_local_id(0); \n \
    int ty = get_local_id(1); \n \
    int aBegin = wA * BLOCK_SIZE * by; \n \
    int aEnd   = aBegin + wA - 1; \n \
    int aStep  = BLOCK_SIZE; \n \
    int bBegin = BLOCK_SIZE * bx; \n \
    int bStep  = BLOCK_SIZE * wB; \n \
    float Csub = 0; \n \
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) { \n \
        __local float As[BLOCK_SIZE][BLOCK_SIZE]; \n \
        __local float Bs[BLOCK_SIZE][BLOCK_SIZE]; \n \
        As[ty][tx] = A[a + wA * ty + tx]; \n \
        Bs[ty][tx] = B[b + wB * ty + tx]; \n \
        barrier(CLK_GLOBAL_MEM_FENCE); \n \
        for (int k = 0; k < BLOCK_SIZE; ++k) { \n \
            Csub += As[ty][k] * Bs[k][tx]; \n \
        } \n \
        barrier(CLK_GLOBAL_MEM_FENCE); \n \
    } \n \
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx; \n \
    C[c + wB * ty + tx] = Csub; \n \
}\n \
";
