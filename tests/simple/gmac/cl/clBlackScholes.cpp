#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "gmac/cl.h"

#include "utils.h"
#include "debug.h"

#include "clBlackScholesKernel.cl"

#define GROUP_SIZE 256
#define S_LOWER_LIMIT 10.0f
#define S_UPPER_LIMIT 100.0f
#define K_LOWER_LIMIT 10.0f
#define K_UPPER_LIMIT 100.0f
#define T_LOWER_LIMIT 1.0f
#define T_UPPER_LIMIT 10.0f
#define R_LOWER_LIMIT 0.01f
#define R_UPPER_LIMIT 0.05f
#define SIGMA_LOWER_LIMIT 0.01f
#define SIGMA_UPPER_LIMIT 0.10f

float
phi(float X)
{
    float y, absX, t;

    // the coeffs
    const float c1 =  0.319381530f;
    const float c2 = -0.356563782f;
    const float c3 =  1.781477937f;
    const float c4 = -1.821255978f;
    const float c5 =  1.330274429f;

    const float oneBySqrt2pi = 0.398942280f;

    absX = fabs(X);
    t = 1.0f / (1.0f + 0.2316419f * absX);

    y = 1.0f - oneBySqrt2pi * exp(-X * X / 2.0f) *
        t * (c1 +
             t * (c2 +
                  t * (c3 +
                       t * (c4 + t * c5))));

    return (X < 0) ? (1.0f - y) : y;
}

void
blackScholesCPU(cl_float *randArray, cl_int width, cl_int height, cl_float *hostCallPrice, cl_float *hostPutPrice)
{
    int y;
    for (y = 0; y < width * height * 4; ++y) {
        float d1, d2;
        float sigmaSqrtT;
        float KexpMinusRT;
        float s = S_LOWER_LIMIT * randArray[y] + S_UPPER_LIMIT * (1.0f - randArray[y]);
        float k = K_LOWER_LIMIT * randArray[y] + K_UPPER_LIMIT * (1.0f - randArray[y]);
        float t = T_LOWER_LIMIT * randArray[y] + T_UPPER_LIMIT * (1.0f - randArray[y]);
        float r = R_LOWER_LIMIT * randArray[y] + R_UPPER_LIMIT * (1.0f - randArray[y]);
        float sigma = SIGMA_LOWER_LIMIT * randArray[y] + SIGMA_UPPER_LIMIT * (1.0f - randArray[y]);

        sigmaSqrtT = sigma * sqrt(t);

        d1 = (log(s / k) + (r + sigma * sigma / 2.0f) * t) / sigmaSqrtT;
        d2 = d1 - sigmaSqrtT;

        KexpMinusRT = k * exp(-r * t);
        hostCallPrice[y] = s * phi(d1) - KexpMinusRT * phi(d2);
        hostPutPrice[y]  = KexpMinusRT * phi(-d2) - s * phi(-d1);
    }
}

int main(int argc, char *argv[])
{
    cl_platform_id platform;
    cl_device_id device;
    cl_int error_code;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;

    gmactime_t s, t;

    cl_uint samples = 256 * 256 * 4;
    size_t blockSizeX = 1;
    size_t blockSizeY = 1;
    cl_float *randArray = NULL;
    cl_float *deviceCallPrice = NULL;
    cl_float *devicePutPrice = NULL;
    cl_float *hostCallPrice = NULL;
    cl_float *hostPutPrice = NULL;
    size_t *maxWorkItemSizes = NULL;
    cl_uint width = 64;
    cl_uint height = 64;
    size_t kernelWorkGroupSize;
    cl_uint maxDimensions;
    size_t maxWorkGroupSize;

    /* Calculate width and height from samples */
    samples = samples / 4;
    samples = (samples / GROUP_SIZE)? (samples / GROUP_SIZE) * GROUP_SIZE: GROUP_SIZE;

    cl_uint tempVar1 = (cl_uint)sqrt((double)samples);
    tempVar1 = (tempVar1 / GROUP_SIZE)? (tempVar1 / GROUP_SIZE) * GROUP_SIZE: GROUP_SIZE;
    samples = tempVar1 * tempVar1;

    width = tempVar1;
    height = width;

    error_code = clGetPlatformIDs(1, &platform, NULL);
    assert(error_code == CL_SUCCESS);
    error_code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    assert(error_code == CL_SUCCESS);
    context = clCreateContext(0, 1, &device, NULL, NULL, &error_code);
    assert(error_code == CL_SUCCESS);
    command_queue = clCreateCommandQueue(context, device, 0, &error_code);
    assert(error_code == CL_SUCCESS);

    /* Get Device specific Information */
    error_code = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), (void*)&maxWorkGroupSize, NULL);
    assert(error_code == CL_SUCCESS);
    error_code = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), (void*)&maxDimensions, NULL);
    assert(error_code == CL_SUCCESS);
    maxWorkItemSizes = (size_t*)malloc(maxDimensions * sizeof(size_t));
    if(maxWorkItemSizes == NULL)
        return 0;
    error_code = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * maxDimensions, (void*)maxWorkItemSizes, NULL);
    assert(error_code == CL_SUCCESS);

    program = clCreateProgramWithSource(context, 1, &code, NULL, &error_code);
    assert(error_code == CL_SUCCESS);
    error_code = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    assert(error_code == CL_SUCCESS);
    kernel = clCreateKernel(program, "blackScholes", &error_code);
    assert(error_code == CL_SUCCESS);

    /* Check if blockSize exceeds group-size returned by kernel */
    error_code = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernelWorkGroupSize, 0);
    assert(error_code == CL_SUCCESS);
    // Calculte 2D block size according to required work-group size by kernel
    kernelWorkGroupSize = kernelWorkGroupSize > GROUP_SIZE ? GROUP_SIZE : kernelWorkGroupSize;
    while((blockSizeX * blockSizeY) < kernelWorkGroupSize) {
        bool next = false;
        if(2 * blockSizeX * blockSizeY <= kernelWorkGroupSize) {
            blockSizeX <<= 1;
            next = true;
        }
        if(2 * blockSizeX * blockSizeY <= kernelWorkGroupSize) {
            next = true;
            blockSizeY <<= 1;
        }

        // Break if no if statement is executed
        if(next == false)
            break;
    }

    getTime(&s);
    // Alloc & init input data
    error_code = clMalloc(command_queue, (void **)&randArray, width * height * sizeof(cl_float4));
	assert(error_code == CL_SUCCESS);
    error_code = clMalloc(command_queue, (void **)&deviceCallPrice, width * height * sizeof(cl_float4));
	assert(error_code == CL_SUCCESS);
    error_code = clMalloc(command_queue, (void **)&devicePutPrice, width * height * sizeof(cl_float4));
	assert (error_code == CL_SUCCESS);
    hostCallPrice = (cl_float*)malloc(width * height * sizeof(cl_float4));
    if(hostCallPrice == NULL)
        return 0;
    hostPutPrice = (cl_float*)malloc(width * height * sizeof(cl_float4));
    if(hostPutPrice == NULL)
        return 0;
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    getTime(&s);
    // random initialisation of input
    for(cl_uint i = 0; i < width * height * 4; i++)
        randArray[i] = (float)rand() / (float)RAND_MAX;

    memset(deviceCallPrice, 0, width * height * sizeof(cl_float4));
    memset(devicePutPrice, 0, width * height * sizeof(cl_float4));
    memset(hostCallPrice, 0, width * height * sizeof(cl_float4));
    memset(hostPutPrice, 0, width * height * sizeof(cl_float4));
    getTime(&t);
    printTime(&s, &t, "Init: ", "\n");

    // Call the kernel
    getTime(&s);
    size_t globalThreads[2] = {width, height};
    size_t localThreads[2] = {blockSizeX, blockSizeY};
    if(localThreads[0] > maxWorkItemSizes[0] ||
       localThreads[1] > maxWorkItemSizes[1] ||
       (size_t)blockSizeX * blockSizeY > maxWorkGroupSize) {
        printf("Unsupported: Device does not support \n requested number of work items.");
        free(maxWorkItemSizes);
        maxWorkItemSizes = NULL;
        return 0;
    }

    cl_mem randArray_device = clGetBuffer(context, randArray);
    cl_mem deviceCallPrice_device = clGetBuffer(context, deviceCallPrice);
    cl_mem devicePutPrice_device = clGetBuffer(context, devicePutPrice);
    error_code = clSetKernelArg(kernel, 0, sizeof(cl_mem), &randArray_device);
	assert(error_code == CL_SUCCESS);
    error_code = clSetKernelArg(kernel, 1, sizeof(width), &width);
	assert(error_code == CL_SUCCESS);
    error_code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &deviceCallPrice_device);
	assert(error_code == CL_SUCCESS);
    error_code = clSetKernelArg(kernel, 3, sizeof(cl_mem), &devicePutPrice_device);
	assert(error_code == CL_SUCCESS);

    error_code = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalThreads, localThreads, 0, NULL, NULL);
	assert(error_code == CL_SUCCESS);
    error_code = clFinish(command_queue);
	assert(error_code == CL_SUCCESS);
    getTime(&t);
    printTime(&s, &t, "Run: ", "\n");

    getTime(&s);
    printf("deviceCallPrice£º\n");
    for(cl_uint i = 0; i < width; i++) {
        printf("%f ", deviceCallPrice[i]);
    }
    printf("\ndevicePutPrice£º\n");
    for(cl_uint i = 0; i < width; i++) {
        printf("%f ", devicePutPrice[i]);
    }

    blackScholesCPU(randArray, width, height, hostCallPrice, hostPutPrice);
    printf("\nhostCallPrice£º\n");
    for(cl_uint i = 0; i < width; i++) {
        printf("%f ", hostCallPrice[i]);
    }
    printf("\nhostPutPrice£º\n");
    for(cl_uint i = 0; i < width; i++) {
        printf("%f ", hostPutPrice[i]);
    }
    getTime(&t);
    printTime(&s, &t, "Print: ", "\n");

    getTime(&s);
    float error = 0.0f;
    float ref = 0.0f;
    bool callPriceResult = true;
    bool putPriceResult = true;
    float normRef;

    for(cl_uint i = 1; i < width * height * 4; ++i) {
        float diff = hostCallPrice[i] - deviceCallPrice[i];
        error += diff * diff;
        ref += hostCallPrice[i] * deviceCallPrice[i];
    }

    normRef =::sqrtf((float) ref);
    if (::fabs((float) ref) < 1e-7f) {
        callPriceResult = false;
    }
    if(callPriceResult) {
        float normError = ::sqrtf((float) error);
        error = normError / normRef;
        callPriceResult = error < 1e-6f;
    }


    for(cl_uint i = 1; i < width * height * 4; ++i) {
        float diff = hostPutPrice[i] - devicePutPrice[i];
        error += diff * diff;
        ref += hostPutPrice[i] * devicePutPrice[i];
    }

    normRef =::sqrtf((float) ref);
    if (::fabs((float) ref) < 1e-7f) {
        putPriceResult = false;
    }
    if(putPriceResult) {
        float normError = ::sqrtf((float) error);
        error = normError / normRef;
        putPriceResult = error < 1e-4f;
    }

    if(!(callPriceResult ? (putPriceResult ? true : false) : false)) {
        printf("Failed!\n");
    } else {
        printf("Passed!\n");
    }
    getTime(&t);
    printTime(&s, &t, "Check: ", "\n");

    getTime(&s);
	/* Release memory */
    free(hostPutPrice);
    hostPutPrice = NULL;
    free(hostCallPrice);
    hostCallPrice = NULL;
    error_code = clFree(command_queue, devicePutPrice);
	assert(error_code == CL_SUCCESS);
    error_code = clFree(command_queue, deviceCallPrice);
	assert(error_code == CL_SUCCESS);
    error_code = clFree(command_queue, randArray);
	assert(error_code == CL_SUCCESS);
    free(maxWorkItemSizes);
    maxWorkItemSizes = NULL;

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

    return 0;
}
