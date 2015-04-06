#include <cassert>

#include <gmac.h>
#include <gmac/vm.h>

const unsigned int THREADS = 256;
const unsigned int BLOCKS  = 32768;

const unsigned int STRIDE = 4;

template <unsigned LOADS, unsigned STORES, unsigned DISTANCE_LD_ST, unsigned DISTANCE_ST_ST>
__global__
void compute(const int * globalIn, int * globalOut)
{
    unsigned index = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ int shared[THREADS * STRIDE];

    if (LOADS > 1) {
        #pragma unroll
        for (unsigned i = 0; i < LOADS; i++) {
            shared[threadIdx.x + (i % STRIDE) * THREADS] = globalIn[index + (i % STRIDE) * THREADS];
        }
    } else if (LOADS == 1) {
        shared[threadIdx.x] = globalIn[index];
    }

    if (DISTANCE_LD_ST > 1) {
        #pragma unroll
        for (unsigned i = 0; i < DISTANCE_LD_ST; i++) {
            shared[threadIdx.x + (i % STRIDE) * THREADS] = shared[threadIdx.x + ((i + 1) % STRIDE) * THREADS];
        }
    } else if (DISTANCE_LD_ST == 1) {
        shared[threadIdx.x] = shared[threadIdx.x + THREADS];
    }

    if (STORES > 2) {
        unsigned i;
        #pragma unroll
        for (i = 0; i < STORES - 1; i++) {
            __globalSt(&globalOut[index + (i % STRIDE) * THREADS], shared[threadIdx.x + (i % STRIDE) * THREADS]);

            if (DISTANCE_ST_ST > 1) {
                #pragma unroll
                for (unsigned j = 0; j < DISTANCE_ST_ST; j++) {
                    shared[threadIdx.x + (j % STRIDE) * THREADS] = shared[threadIdx.x + (j % STRIDE) * THREADS];
                }
            } else if (DISTANCE_ST_ST == 1) {
                shared[threadIdx.x + (i % STRIDE) * THREADS] = shared[threadIdx.x + ((i + 1) % STRIDE) * THREADS];
            }
        }
        // Perform last store outside the loop
        __globalSt(&globalOut[index + (i % STRIDE) * THREADS], shared[threadIdx.x + (i % STRIDE) * THREADS]);
    } else if (STORES == 2) {
        __globalSt(&globalOut[index], shared[threadIdx.x]);

        if (DISTANCE_ST_ST > 1) {
            #pragma unroll
            for (unsigned j = 0; j < DISTANCE_ST_ST; j++) {
                shared[threadIdx.x + (j % STRIDE) * THREADS] = shared[threadIdx.x + (j % STRIDE) * THREADS];
            }
        } else if (DISTANCE_ST_ST == 1) {
            shared[threadIdx.x] = shared[threadIdx.x + THREADS];
        }
        __globalSt(&globalOut[index + THREADS], shared[threadIdx.x + THREADS]);
    } else if (STORES == 1) {
        __globalSt(&globalOut[index], shared[threadIdx.x]);
    }
}

#define REPS 10

template<unsigned A, unsigned B, unsigned C, unsigned D>
void
doCompute4(const int * in, int * out)
{
    gmacError_t ret;
    for (unsigned i = 0; i < REPS; i++) {
        compute<A, B, C, D>   <<<BLOCKS, THREADS>>>(gmacPtr(in), gmacPtr(out));
        ret = gmacThreadSynchronize();
        assert(ret == gmacSuccess);
    }
}

template<unsigned A, unsigned B, unsigned C>
void
doCompute3(const int * in, int * out)
{
    doCompute4<A, B, C, 0>(in, out);  // 1 
    doCompute4<A, B, C, 1>(in, out);  // 2 
    doCompute4<A, B, C, 2>(in, out);  // 3 
    doCompute4<A, B, C, 4>(in, out);  // 4 
    doCompute4<A, B, C, 8>(in, out);  // 5 
    doCompute4<A, B, C, 16>(in, out); // 6
    doCompute4<A, B, C, 32>(in, out); // 7
    doCompute4<A, B, C, 64>(in, out); // 8
}

template<unsigned A, unsigned B>
void
doCompute2(const int * in, int * out)
{
    doCompute3<A, B, 0>(in, out);  // 1 
    doCompute3<A, B, 1>(in, out);  // 2 
    doCompute3<A, B, 2>(in, out);  // 3 
    doCompute3<A, B, 4>(in, out);  // 4 
    doCompute3<A, B, 8>(in, out);  // 5 
    doCompute3<A, B, 16>(in, out); // 6
    doCompute3<A, B, 32>(in, out); // 7
    doCompute3<A, B, 64>(in, out); // 8
}

template<unsigned A>
void
doCompute1(const int * in, int * out)
{
    doCompute2<A, 1>(in, out);
    doCompute2<A, 2>(in, out);
    doCompute2<A, 4>(in, out);
    doCompute2<A, 8>(in, out);
    doCompute2<A, 16>(in, out);
}

void
doCompute(const int * in, int * out)
{
    doCompute1<1>(in, out);
    doCompute1<2>(in, out);
    doCompute1<4>(in, out);
    doCompute1<8>(in, out);
    doCompute1<16>(in, out);
}

int main(int argc, char * argv[])
{
    gmacError_t ret;
    int * in, * out;

    ret = gmacMalloc((void **) &in, sizeof(int) * THREADS * BLOCKS * STRIDE);
    assert(ret == cudaSuccess);
    ret = gmacMalloc((void **) &out, sizeof(int) * THREADS * BLOCKS * STRIDE);
    assert(ret == cudaSuccess);

    doCompute(in, out);

    assert(ret == cudaSuccess);

    return 0;
}

