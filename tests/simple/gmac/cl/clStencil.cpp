#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "utils.h"
#include "debug.h"
#include "barrier.h"

#include "clStencilCommon.h"


int main(int argc, char *argv[])
{
	setParam<unsigned>(&dimRealElems, dimRealElemsStr, dimRealElemsDefault);
    assert(dimRealElems % 32 == 0);

    error_code = clGetPlatformIDs(1, &platform, NULL);
    assert(error_code == CL_SUCCESS);
    error_code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    assert(error_code == CL_SUCCESS);
    context = clCreateContext(0, 1, &device, NULL, NULL, &error_code);
    assert(error_code == CL_SUCCESS);
    command_queue = clCreateCommandQueue(context, device, 0, &error_code);
    assert(error_code == CL_SUCCESS);
    program = clCreateProgramWithSource(context, 1, &stencilCode, NULL, &error_code);
    assert(error_code == CL_SUCCESS);
    error_code = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    assert(error_code == CL_SUCCESS);
    kernel = clCreateKernel(program, "kernelStencil", &error_code);
    assert(error_code == CL_SUCCESS);

    dimElems = dimRealElems + 2 * STENCIL;

    JobDescriptor * descriptor = new JobDescriptor();
    descriptor->gpus  = 1;
    descriptor->gpuId = 1;

    descriptor->prev = NULL;
    descriptor->next = NULL;

    descriptor->dimRealElems = dimRealElems;
    descriptor->dimElems     = dimElems;
    descriptor->slices       = dimRealElems;

    do_stencil((void *) descriptor);

    delete descriptor;

    return 0;
}
