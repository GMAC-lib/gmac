#ifndef GMAC_STENCIL_COMMON_H_
#define GMAC_STENCIL_COMMON_H_

#include <cassert>
#include <cstring>
#include "gmac/cl.h"

#define STENCIL 4

static const char *stencilCode = "\
\
#define STENCIL 4 \n\
\
__constant \
float devC00; \n\
__constant \
float devZ1; \n\
__constant \
float devZ2; \n\
__constant \
float devZ3; \n\
__constant \
float devZ4; \n\
__constant \
float devX1; \n\
__constant \
float devX2; \n\
__constant \
float devX3; \n\
__constant \
float devX4; \n\
__constant \
float devY1; \n\
__constant \
float devY2; \n\
__constant \
float devY3; \n\
__constant \
float devY4; \n\
\
__kernel \
void \
kernelStencil(__global const float * u2,\
              __global float * u3,\
              __global const float * v,\
              float dt2,\
              unsigned dimZ,\
              unsigned dimRealZ,\
              unsigned dimZX,\
              unsigned dimRealZX,\
              unsigned slices)\
{\n\
    __local float s_data[8 + 2 * STENCIL][32 + 2 * STENCIL]; \n\
\
    unsigned ix = get_global_id(0); \n\
    unsigned iy = get_global_id(1); \n\
\
    unsigned tx = get_local_id(0) + STENCIL; \n\
    unsigned ty = get_local_id(1) + STENCIL; \n\
\
    int index     = (iy * dimZ     + ix) + 3 * dimZX; \n\
    int realIndex = (iy * dimRealZ + ix); \n\
\
    unsigned TILE_OFFSET_LINE = 32 + 2 * STENCIL; \n\
\
    float4 front; \n\
    float4 back; \n\
\
    float current; \n\
\
    back.z  = u2[index - 3 * dimZX]; \n\
    back.y  = u2[index - 2 * dimZX]; \n\
    back.x  = u2[index - 1 * dimZX]; \n\
    current = u2[index            ]; \n\
    front.w = u2[index + 1 * dimZX]; \n\
    front.z = u2[index + 2 * dimZX]; \n\
    front.y = u2[index + 3 * dimZX]; \n\
    front.x = u2[index + 4 * dimZX]; \n\
\
    for (int k = 0; k < (slices - 2 * STENCIL); k++) { \n\
        float tmpU2 = u2[index + ((STENCIL + 1) * dimZX)]; \n\
        index += dimZX; \n\
\
        back.z = back.y; \n\
        back.y = back.x; \n\
        back.x = current; \n\
        current = front.w; \n\
        front.w = front.z; \n\
        front.z = front.y; \n\
        front.y = front.x; \n\
        front.x = tmpU2; \n\
\n\
        barrier(CLK_LOCAL_MEM_FENCE); \n\
\n\
        if (get_local_id(0) < STENCIL) { \n\
            s_data[ty][get_local_id(0)] = u2[index - STENCIL]; \n\
            s_data[ty][tx + 32] = u2[index + 32]; \n\
        }\n\
\
        if (get_local_id(1) < STENCIL) { \n\
            s_data[get_local_id(1)][tx] = u2[index - STENCIL * dimZ]; \n\
            s_data[ty + 8][tx]             = u2[index + 8 * dimZ]; \n\
        }\n\
\n\
        s_data[ty][tx] = current; \n\
        barrier(CLK_LOCAL_MEM_FENCE); \n\
        float tmp  = v[realIndex]; \n\
        float tmp1 = u3[index]; \n\
\n\
        float div  = \
               devX4 * (s_data[ty - 4][tx] + s_data[ty + 4][tx]); \n\
        div += devC00 * current; \n\
        div += devX3 * (s_data[ty - 3][tx] + s_data[ty + 3][tx]); \n\
        div += devX2 * (s_data[ty - 2][tx] + s_data[ty + 2][tx]); \n\
        div += devX1 * (s_data[ty - 1][tx] + s_data[ty + 1][tx]); \n\
        div += devY4 * (front.x + back.w); \n\
        div += devZ4 * (s_data[ty][tx - 4] + s_data[ty][tx + 4]); \n\
        div += devY3 * (front.y + back.z); \n\
        div += devZ3 * (s_data[ty][tx - 3] + s_data[ty][tx + 3]); \n\
        div += devY2 * (front.z + back.y); \n\
        div += devZ2 * (s_data[ty][tx - 2] + s_data[ty][tx + 2]); \n\
        div += devY1 * (front.w + back.x); \n\
        div += devZ1 * (s_data[ty][tx - 1] + s_data[ty][tx + 1]); \n\
\n\
        div = tmp * tmp * div; \n\
        div = dt2 * div + current + current - tmp1; \n\
        u3[index] = div; \n\
\n\
        realIndex += dimRealZX; \n\
    } \n\
}";

#define VELOCITY 2000

barrier_t barrier;

struct JobDescriptor {
    const static int DEFAULT_DIM = 32;
    int gpus;
    int gpuId;

    struct JobDescriptor * prev;
    struct JobDescriptor * next;

    float * u3;
    float * u2;

    unsigned dimRealElems;
    unsigned dimElems;
    unsigned slices;

    unsigned sliceElems()
    {
        return dimElems * dimElems;
    }

    unsigned sliceRealElems()
    {
        return dimRealElems * dimRealElems;
    }

    unsigned elems()
    {
        return dimElems * dimElems * (slices + 2 * STENCIL);
    }

    unsigned realElems()
    {
        return dimRealElems * dimRealElems * slices;
    }

    unsigned size()
    {
        return dimElems * dimElems * (slices + 2 * STENCIL) * sizeof(float);
    }

    unsigned realSize()
    {
        return dimRealElems * dimRealElems * slices * sizeof(float);
    }


};

static cl_platform_id platform;
static cl_device_id device;
static cl_int error_code;
static cl_context context;
static cl_command_queue command_queue;
static cl_program program;
static cl_kernel kernel;

#define ITERATIONS 50

void *
do_stencil(void * ptr)
{
    JobDescriptor * descr = (JobDescriptor *) ptr;

	float * v = NULL;
	gmactime_t s, t;

	getTime(&s);

	// Alloc 3 volumes for 2-degree time integration
	error_code = clMalloc(command_queue, (void **)&descr->u2, descr->size());
	assert(error_code == CL_SUCCESS);
	memset(descr->u2, 0, descr->size());
	error_code = clMalloc(command_queue, (void **)&descr->u3, descr->size());
	assert(error_code == CL_SUCCESS);
    memset(descr->u3, 0, descr->size());

    error_code = clMalloc(command_queue, (void **) &v, descr->realSize());
	assert(error_code == CL_SUCCESS);

    for (size_t k = 0; k < descr->slices; k++) {        
        for (size_t j = 0; j < descr->dimRealElems; j++) {        
            for (size_t i = 0; i < descr->dimRealElems; i++) {        
                size_t iter = k * descr->sliceRealElems() + j * descr->dimRealElems + i;
                v[iter] = VELOCITY;
            }
        }
    }

    if (descr->gpus > 1) {
        barrier_wait(&barrier);
    }

	getTime(&t);
	printTime(&s, &t, "Alloc: ", "\n");

    size_t localSize[2] = {32, 8};
    size_t globalSize[2];
    globalSize[0] = descr->dimElems - 2 * STENCIL;
    globalSize[1] = descr->dimElems - 2 * STENCIL;

	getTime(&s);

    cl_mem tmpMem = clGetBuffer(context, v);
    error_code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &tmpMem);
	assert(error_code == CL_SUCCESS);
    float dt2 = 0.08f;
    error_code = clSetKernelArg(kernel, 3, sizeof(dt2), &dt2);
	assert(error_code == CL_SUCCESS);
    error_code = clSetKernelArg(kernel, 4, sizeof(descr->dimElems), &descr->dimElems);
	assert(error_code == CL_SUCCESS);
    error_code = clSetKernelArg(kernel, 5, sizeof(descr->dimRealElems), &descr->dimRealElems);
	assert(error_code== CL_SUCCESS);
    unsigned intTmp = descr->sliceElems();
    error_code = clSetKernelArg(kernel, 6, sizeof(intTmp), &intTmp);
	assert(error_code == CL_SUCCESS);
    intTmp = descr->sliceRealElems();
    error_code = clSetKernelArg(kernel, 7, sizeof(intTmp), &intTmp);
	assert(error_code == CL_SUCCESS);
    error_code = clSetKernelArg(kernel, 8, sizeof(descr->slices), &descr->slices);
	assert(error_code == CL_SUCCESS);


    for (uint32_t i = 1; i <= ITERATIONS; i++) {
        float * tmp;
        
        // Call the kernel
        tmpMem = clGetBuffer(context, descr->u2);
        error_code = clSetKernelArg(kernel, 0, sizeof(cl_mem), &tmpMem);
		assert(error_code == CL_SUCCESS);
        tmpMem = clGetBuffer(context, descr->u3);                            
        error_code = clSetKernelArg(kernel, 1, sizeof(cl_mem), &tmpMem);
		assert(error_code == CL_SUCCESS);
        
#if 0
        error_code = __oclKernelConfigure(&kernel, 2, NULL, globalSize, localSize);
		assert(error_code == CL_SUCCESS);
        oclError_t ret;
        ret = __oclKernelLaunch(&kernel);
        assert(ret == CL_SUCCESS);
#endif

        error_code = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
		assert(error_code == CL_SUCCESS);

#if 0
        if(descr->gpus > 1) {
            error_code = clFinish(command_queue);
			assert(error_code == CL_SUCCESS);
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

            barrier_wait(&barrier);
        }
#endif

        tmp = descr->u3;
        descr->u3 = descr->u2;
        descr->u2 = tmp;
    }

    error_code = clFinish(command_queue);
	assert(error_code == CL_SUCCESS);
    if(descr->gpus > 1) {
        barrier_wait(&barrier);
    }

	getTime(&t);
	printTime(&s, &t, "Run: ", "\n");

    getTime(&s);
	/* Release memory */
	error_code = clFree(command_queue, descr->u2);
	assert(error_code == CL_SUCCESS);
	error_code = clFree(command_queue, descr->u3);
	assert(error_code == CL_SUCCESS);
	error_code = clFree(command_queue, v);
	assert(error_code == CL_SUCCESS);

	/* Release OpenCL resources */
	error_code = clReleaseKernel(kernel);
	assert(error_code == CL_SUCCESS);
	error_code = clReleaseProgram(program);
	assert(error_code == CL_SUCCESS);
	error_code = clReleaseCommandQueue(command_queue);
	assert(error_code == CL_SUCCESS);
	error_code = clReleaseContext(context);
	assert(error_code == CL_SUCCESS);
    getTime(&t);
    printTime(&s, &t, "Free: ", "\n");

    return NULL;
}

const char * dimRealElemsStr = "GMAC_STENCIL_DIM_ELEMS";
const unsigned dimRealElemsDefault = 352;

static unsigned dimElems     = 0;
static unsigned dimRealElems = 0;

#endif
