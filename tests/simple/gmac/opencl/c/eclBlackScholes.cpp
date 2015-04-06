#include <cstdio>
#include <cstdlib>

#include "gmac/opencl.h"

#include "utils.h"
#include "debug.h"

#include "../eclBlackScholesKernel.cl"

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
	gmactime_t s, t, S, T;

	cl_uint samples = 256 * 256 * 4;
	size_t blockSizeX = 1;
	size_t blockSizeY = 1;
	cl_float *randArray = NULL;
	cl_float *deviceCallPrice = NULL;
	cl_float *devicePutPrice = NULL;
	cl_float *hostCallPrice = NULL;
	cl_float *hostPutPrice = NULL;
	cl_uint width = 64;
	cl_uint height = 64;

	/* Calculate width and height from samples */
	samples = samples / 4;
	samples = (samples / GROUP_SIZE)? (samples / GROUP_SIZE) * GROUP_SIZE: GROUP_SIZE;

	cl_uint tempVar1 = (cl_uint)sqrt((double)samples);
	tempVar1 = (tempVar1 / GROUP_SIZE)? (tempVar1 / GROUP_SIZE) * GROUP_SIZE: GROUP_SIZE;
	samples = tempVar1 * tempVar1;

	width = tempVar1;
	height = width;

	assert(eclCompileSource(code) == eclSuccess);
	ecl_accelerator_info info;
	assert(eclGetAcceleratorInfo(0, &info) == eclSuccess);

	getTime(&s);
	// Alloc & init input data
	assert(eclMalloc((void **)&randArray, width * height * sizeof(cl_float4)) == eclSuccess);
	assert(eclMalloc((void **)&deviceCallPrice, width * height * sizeof(cl_float4)) == eclSuccess);
	assert(eclMalloc((void **)&devicePutPrice, width * height * sizeof(cl_float4)) == eclSuccess);
	hostCallPrice = (cl_float*)malloc(width * height * sizeof(cl_float4));
	if(hostCallPrice == NULL)
		return 0;
	hostPutPrice = (cl_float*)malloc(width * height * sizeof(cl_float4));
	if(hostPutPrice == NULL)
		return 0;
	getTime(&t);
	printTime(&s, &t, "Alloc: ", "\n");

	getTime(&S);
	getTime(&s);
	// random initialisation of input
	for(cl_uint i = 0; i < width * height * 4; i++)
		randArray[i] = (float)rand() / (float)RAND_MAX;

	eclMemset(deviceCallPrice, 0, width * height * sizeof(cl_float4));
	eclMemset(devicePutPrice, 0, width * height * sizeof(cl_float4));
	eclMemset(hostCallPrice, 0, width * height * sizeof(cl_float4));
	eclMemset(hostPutPrice, 0, width * height * sizeof(cl_float4));
	getTime(&t);
	printTime(&s, &t, "Init: ", "\n");

	// Call the kernel
	getTime(&s);
	size_t globalThreads[2] = {width, height};
	size_t localThreads[2] = {blockSizeX, blockSizeY};
	ecl_kernel kernel;
	assert(eclGetKernel("blackScholes", &kernel) == eclSuccess);
	assert(eclSetKernelArgPtr(kernel, 0, randArray) == eclSuccess);
	assert(eclSetKernelArg(kernel, 1, sizeof(width), &width) == eclSuccess);
	assert(eclSetKernelArgPtr(kernel, 2, deviceCallPrice) == eclSuccess);
	assert(eclSetKernelArgPtr(kernel, 3, devicePutPrice) == eclSuccess);
	assert(eclCallNDRange(kernel, 2, NULL, globalThreads, localThreads) == eclSuccess);

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
	getTime(&T);
	printTime(&S, &T, "Total: ", "\n");

	getTime(&s);
	free(hostPutPrice);
	hostPutPrice = NULL;
	free(hostCallPrice);
	hostCallPrice = NULL;

	eclReleaseKernel(kernel);

	eclFree(devicePutPrice);
	eclFree(deviceCallPrice);
	eclFree(randArray);
	getTime(&t);
	printTime(&s, &t, "Free: ", "\n");
	return 0;
}
