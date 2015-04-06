#ifndef GMAC_STENCIL_COMMON
#define GMAC_STENCIL_COMMON

#include <gmac/vm.h>

#define STENCIL 4

__constant__
float devC00;
__constant__
float devZ1;
__constant__
float devZ2;
__constant__
float devZ3;
__constant__
float devZ4;
__constant__
float devX1;
__constant__
float devX2;
__constant__
float devX3;
__constant__
float devX4;
__constant__
float devY1;
__constant__
float devY2;
__constant__
float devY3;
__constant__
float devY4;


template <uint32_t STENCIL_TILE_XSIZE, uint32_t STENCIL_TILE_YSIZE>
__global__
void
kernelStencil(const float * u2,
              float * u3,
              const float * v,
              const float dt2,
              const size_t dimZ,
              const size_t dimRealZ,
              const size_t dimZX,
              const size_t dimRealZX,
              const size_t slices)
{
    __shared__
        __align__(0x10)
        float s_data[(STENCIL_TILE_YSIZE + 2 * STENCIL) * (STENCIL_TILE_XSIZE + 2 * STENCIL)];

    uint32_t ix = (blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t iy = (blockIdx.y * blockDim.y + threadIdx.y);

    uint32_t tx = threadIdx.x + STENCIL; // thread’s x-index into corresponding shared memory tile (adjusted for halos)
    uint32_t ty = threadIdx.y + STENCIL; // thread’s y-index into corresponding shared memory tile (adjusted for halos)

    int32_t index     = (iy * dimZ     + ix) + 3 * dimZX; // index for reading input
    int32_t realIndex = (iy * dimRealZ + ix); // index for reading/writing from/to structures without ghost area

#define TILE_OFFSET_LINE (STENCIL_TILE_XSIZE + 2 * STENCIL)
#define SH(x,y) ((y) * TILE_OFFSET_LINE + x)
#define SH_X(off) (uint32_t(int32_t(SH(tx, ty)) + (off)))
#define SH_Y(off) (uint32_t(int32_t(SH(tx, ty)) + int32_t(off) * int32_t(TILE_OFFSET_LINE)))

    float4 front;
    float4 back;

    float current;

    // fill the "in-front" and "behind" data
    back.z  = u2[index - 3 * dimZX];
    back.y  = u2[index - 2 * dimZX];
    back.x  = u2[index - 1 * dimZX];
    current = u2[index            ];
    front.w = u2[index + 1 * dimZX];
    front.z = u2[index + 2 * dimZX];
    front.y = u2[index + 3 * dimZX];
    front.x = u2[index + 4 * dimZX];

    //int signY = (threadIdx.y - STENCIL) >> 31;
    for (int k = 0; k < (slices - 2 * STENCIL); k++) {
        float tmpU2 = u2[index + ((STENCIL + 1) * dimZX)];
        index += dimZX;

        //////////////////////////////////////////
        // advance the slice (move the thread-front)
        back.w = back.z;
        back.z = back.y;
        back.y = back.x;
        back.x = current;
        current = front.w;
        front.w = front.z;
        front.z = front.y;
        front.y = front.x;
        front.x = tmpU2;

        __syncthreads();

        /////////////////////////////////////////
        // update the data slice in smem
        //s_data[SH(tx + (STENCIL * signX), ty)] = u2[index + STENCIL * signX];
        //s_data[SH(tx, ty + signY * STENCIL)] = u2[index + dimZ) * signY * STENCIL];

        if (threadIdx.x < STENCIL) { // halo left/right
            s_data[SH(threadIdx.x, ty)            ] = u2[index - STENCIL];
            s_data[SH(tx + STENCIL_TILE_XSIZE, ty)] = u2[index + STENCIL_TILE_XSIZE];
        }
        __syncthreads();
        if (threadIdx.y < STENCIL) { // halo above/below
            s_data[SH(tx, threadIdx.y)            ] = u2[index - STENCIL            * dimZ];
            s_data[SH(tx, ty + STENCIL_TILE_YSIZE)] = u2[index + STENCIL_TILE_YSIZE * dimZ];
        }

        /////////////////////////////////////////
        // compute the output value
        s_data[SH(tx, ty)] = current;
        __syncthreads();
        float tmp  = v[realIndex];
        float tmp1 = u3[index];

        float div  =
              devX4 * (s_data[SH_Y(-4)] + s_data[SH_Y(4)]);
        div += devC00 * current;
        div += devX3 * (s_data[SH_Y(-3)] + s_data[SH_Y(3)]);
        div += devX2 * (s_data[SH_Y(-2)] + s_data[SH_Y(2)]);
        div += devX1 * (s_data[SH_Y(-1)] + s_data[SH_Y(1)]);
        div += devY4 * (front.x + back.w);
        div += devZ4 * (s_data[SH_X(-4)] + s_data[SH_X(4)]);
        div += devY3 * (front.y + back.z);
        div += devZ3 * (s_data[SH_X(-3)] + s_data[SH_X(3)]);
        div += devY2 * (front.z + back.y);
        div += devZ2 * (s_data[SH_X(-2)] + s_data[SH_X(2)]);
        div += devY1 * (front.w + back.x);
        div += devZ1 * (s_data[SH_X(-1)] + s_data[SH_X(1)]);

        div = tmp * tmp * div;
        div = dt2 * div + current + current - tmp1;
        __globalSt<float>(&u3[index], div);

        realIndex += dimRealZX;
    }
}

#define VELOCITY 2000

//pthread_barrier_t barrier;
barrier_t barrier;

struct JobDescriptor {
    const static int DEFAULT_DIM = 256;
    int gpus;
    int gpuId;

    struct JobDescriptor * prev;
    struct JobDescriptor * next;

    float * u3;
    float * u2;

    size_t dimRealElems;
    size_t dimElems;
    size_t slices;

    size_t sliceElems()
    {
        return dimElems * dimElems;
    }

    size_t sliceRealElems()
    {
        return dimRealElems * dimRealElems;
    }

    size_t elems()
    {
        return dimElems * dimElems * (slices + 2 * STENCIL);
    }

    size_t realElems()
    {
        return dimRealElems * dimRealElems * slices;
    }

    size_t size()
    {
        return dimElems * dimElems * (slices + 2 * STENCIL) * sizeof(float);
    }

    size_t realSize()
    {
        return dimRealElems * dimRealElems * slices * sizeof(float);
    }


};

#define ITERATIONS 50

void *
do_stencil(void * ptr)
{
    JobDescriptor * descr = (JobDescriptor *) ptr;

	float * v = NULL;
	gmactime_t s, t;

	getTime(&s);

	// Alloc 3 volumes for 2-degree time integration
	if(gmacMalloc((void **)&descr->u2, descr->size()) != gmacSuccess)
		CUFATAL();
	gmacMemset(descr->u2, 0, descr->size());
	if(gmacMalloc((void **)&descr->u3, descr->size()) != gmacSuccess)
		CUFATAL();
    gmacMemset(descr->u3, 0, descr->size());

    if(gmacMalloc((void **) &v, descr->realSize()) != gmacSuccess)
		CUFATAL();

    for (unsigned k = 0; k < descr->slices; k++) {        
        for (unsigned j = 0; j < descr->dimRealElems; j++) {        
            for (unsigned i = 0; i < descr->dimRealElems; i++) {        
                size_t iter = k * descr->sliceRealElems() + j * descr->dimRealElems + i;
                v[iter] = VELOCITY;
            }
        }
    }

	getTime(&t);
	printTime(&s, &t, "Alloc: ", "\n");

	dim3 Db(32, 8);
	dim3 Dg(unsigned(descr->dimElems / 32), unsigned(descr->dimElems / 8));
	getTime(&s);
    for (uint32_t i = 1; i <= ITERATIONS; i++) {
        if (i % 10 == 0)
            printf("Iteration: %d\n", i);
        float * tmp;
        // Call the kernel
        kernelStencil<32, 8><<<Dg, Db>>>(gmacPtr(descr->u2 + descr->dimElems * STENCIL + STENCIL),
                                         gmacPtr(descr->u3 + descr->dimElems * STENCIL + STENCIL),
                                         gmacPtr(v),
                                         0.08f,
                                         descr->dimElems, descr->dimRealElems, descr->sliceElems(), descr->sliceRealElems(),
                                         descr->slices);
        //if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();

        if(descr->gpus > 1) {
            //pthread_barrier_wait(&barrier);
            barrier_wait(&barrier);

            // Send data
            if (descr->prev != NULL) {
                gmacMemcpy(descr->prev->u3 + descr->elems() - STENCIL * descr->sliceElems(),
                           descr->u3 + STENCIL * descr->sliceElems(),
                           descr->sliceElems() * STENCIL * sizeof(float));
            }
            if (descr->next != NULL) {
                gmacMemcpy(descr->next->u3,
                           descr->u3 + descr->elems() - 2 * STENCIL * descr->sliceElems(),
                           descr->sliceElems() * STENCIL * sizeof(float));                
            }

            //pthread_barrier_wait(&barrier);
            barrier_wait(&barrier);
        }

        tmp = descr->u3;
        descr->u3 = descr->u2;
        descr->u2 = tmp;
    }

    if(descr->gpus > 1) {
        //pthread_barrier_wait(&barrier);
        barrier_wait(&barrier);
    }

	getTime(&t);
	printTime(&s, &t, "Run: ", "\n");

	gmacFree(descr->u2);
	gmacFree(descr->u3);

	gmacFree(v);

    return NULL;
}

const char * dimRealElemsStr = "GMAC_DIM_REAL_ELEMS";
const size_t dimRealElemsDefault = 352;

static size_t dimElems     = 0;
static size_t dimRealElems = 0;

#endif
