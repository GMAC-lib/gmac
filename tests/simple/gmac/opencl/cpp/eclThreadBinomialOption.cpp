#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "gmac/opencl"

#include "utils.h"
#include "debug.h"

#include "../eclThreadBinomialOptionKernel.cl"

#define VOLATILITY 0.30f
#define RISKFREE 0.02f

static cl_int numSamples = 64;
static cl_int numSteps = 254;
cl_float* randArray = NULL;
cl_float* output = NULL;
cl_float* refOutput = NULL;

int
binomialOptionCPUReference(cl_float *refOutput, cl_float *randArray, cl_int numSamples, cl_int numSteps)
{
	float* stepsArray = (float*)malloc((numSteps + 1) * sizeof(cl_float4));
	if(stepsArray== NULL)
		return 0;
	/* Iterate for all samples */
	for(int bid = 0; bid < numSamples; ++bid) {
		float s[4];
		float x[4];
		float vsdt[4];
		float puByr[4];
		float pdByr[4];
		float optionYears[4];

		float inRand[4];

		for(int i = 0; i < 4; ++i) {
			inRand[i] = randArray[bid + i];
			s[i] = (1.0f - inRand[i]) * 5.0f + inRand[i] * 30.f;
			x[i] = (1.0f - inRand[i]) * 1.0f + inRand[i] * 100.f;
			optionYears[i] = (1.0f - inRand[i]) * 0.25f + inRand[i] * 10.f;
			float dt = optionYears[i] * (1.0f / (float)numSteps);
			vsdt[i] = VOLATILITY * sqrtf(dt);
			float rdt = RISKFREE * dt;
			float r = expf(rdt);
			float rInv = 1.0f / r;
			float u = expf(vsdt[i]);
			float d = 1.0f / u;
			float pu = (r - d)/(u - d);
			float pd = 1.0f - pu;
			puByr[i] = pu * rInv;
			pdByr[i] = pd * rInv;
		}
		// Compute values at expiration date:
		// Call option value at period end is v(t) = s(t) - x
		// If s(t) is greater than x, or zero otherwise...
		// The computation is similar for put options...
		for(int j = 0; j <= numSteps; j++) {
			for(int i = 0; i < 4; ++i) {
				float profit = s[i] * expf(vsdt[i] * (2.0f * j - numSteps)) - x[i];
				stepsArray[j * 4 + i] = profit > 0.0f ? profit : 0.0f;
			}
		}

		//walk backwards up on the binomial tree of depth numSteps
		//Reduce the price step by step
		for(int j = numSteps; j > 0; --j) {
			for(int k = 0; k <= j - 1; ++k) {
				for(int i = 0; i < 4; ++i) {
					stepsArray[k * 4 + i] = pdByr[i] * stepsArray[(k + 1) * 4 + i] + puByr[i] * stepsArray[k * 4 + i];
				}
			}
		}

		//Copy the root to result
		refOutput[bid] = stepsArray[0];
	}

	free(stepsArray);

	return 0;
}

void *binomialOption(void *n)
{
	gmactime_t s, t;
	ecl::error ret = eclSuccess;
	int offset = *(int *)n * numSamples;
	/*cl_float* output_thread = NULL;
	assert(eclMalloc((void **)&output_thread, numSamples * sizeof(cl_float4)) == eclSuccess);*/

	getTime(&s);
	ecl::kernel kernel("binomial_options", ret);
	assert(ret == eclSuccess);
	ret = kernel.setArg(0, numSteps);
	assert(ret == eclSuccess);	
	ret = kernel.setArg(1, randArray);
	assert(ret == eclSuccess);	
	ret = kernel.setArg( 2, output);
	assert(ret == eclSuccess);	
	//assert(eclSetKernelArgPtr(kernel, 2, output_thread) == eclSuccess);
	ret = kernel.setArg(3,  NULL);
	assert(ret == eclSuccess);	
	ret = kernel.setArg(4, NULL);
	assert(ret == eclSuccess);	
	ret = kernel.setArg(5, offset);
	assert(ret == eclSuccess);

	ecl::config globalThreads(numSamples * (numSteps + 1));
	ecl::config localThreads(numSteps + 1);
	ret = kernel.callNDRange(globalThreads, localThreads);
	assert(ret == eclSuccess);	
	getTime(&t);
	printTime(&s, &t, "Run: ", "\n");
	//eclMemcpy(&output[offset], output_thread, numSamples * sizeof(cl_float4));
	//assert(eclFree(output_thread) == eclSuccess);

	return NULL;
}

int main(int argc, char *argv[])
{
	gmactime_t s, t;
	int nIter = 8;
	thread_t *nThread;
	ecl::error ret = eclSuccess;

	// Make numSamples multiple of 4
	numSamples = (numSamples / 4)? (numSamples / 4) * 4: 4;

	ret = ecl::compileSource(code);
	assert(ret == eclSuccess);

	getTime(&s);
	// Alloc & init data
	ret = ecl::malloc((void **)&randArray, nIter * numSamples * sizeof(cl_float4));
	assert(ret == eclSuccess);
	ret = ecl::malloc((void **)&output, nIter * numSamples * sizeof(cl_float4));
	assert(ret == eclSuccess);
	refOutput = (float*)malloc(nIter * numSamples * sizeof(cl_float4));
	if(refOutput == NULL)
		return 0;
	getTime(&t);
	printTime(&s, &t, "Alloc: ", "\n");

	getTime(&s);
	/* random initialisation of input */
	for(int i = 0; i < nIter * numSamples * 4; i++) {
		randArray[i] = (float)rand() / (float)RAND_MAX;
	}
	ecl::memset(output, 0, nIter * numSamples * 4);
	getTime(&t);
	printTime(&s, &t, "Init: ", "\n");

	nThread = (thread_t *)malloc(nIter * sizeof(thread_t));

	getTime(&s);
	int n1 = 0;
	int n2 = 1;
	int n3 = 2;
	int n4 = 3;
	int n5 = 4;
	int n6 = 5;
	int n7 = 6;
	int n8 = 7;
	nThread[0] = thread_create(binomialOption, &n1);
	nThread[1] = thread_create(binomialOption, &n2);
	nThread[2] = thread_create(binomialOption, &n3);
	nThread[3] = thread_create(binomialOption, &n4);
	nThread[4] = thread_create(binomialOption, &n5);
	nThread[5] = thread_create(binomialOption, &n6);
	nThread[6] = thread_create(binomialOption, &n7);
	nThread[7] = thread_create(binomialOption, &n8);

	thread_wait(nThread[0]);
	thread_wait(nThread[1]);
	thread_wait(nThread[2]);
	thread_wait(nThread[3]);
	thread_wait(nThread[4]);
	thread_wait(nThread[5]);
	thread_wait(nThread[6]);
	thread_wait(nThread[7]);

	getTime(&t);
	printTime(&s, &t, "Total: ", "\n");

	printf("\noutput: \n");
	for (int i = 0; i < nIter * numSamples; i++) {
		printf("%f ", output[i]);
	}
	printf("\n");

	getTime(&s);
	bool result = 1;
	binomialOptionCPUReference(refOutput, randArray, nIter * numSamples, numSteps);

	printf("\nrefoutput: \n");
	for (int i = 0; i < nIter * numSamples; i++) {
		printf("%f ", refOutput[i]);
	}
	printf("\n");

	float error = 0.0f;
	float ref = 0.0f;
	for(int i = 1; i < nIter * numSamples; ++i) {
		float diff = output[i] - refOutput[i];
		error += diff * diff;
		ref += output[i] * output[i];
	}

	float normRef =::sqrtf((float) ref);
	if (::fabs((float) ref) < 1e-7f) {
		result = 0;
	}
	if(result) {
		float normError = ::sqrtf((float) error);
		error = normError / normRef;
		result = error < 0.001f;
	}
	if(result)
		printf("\nPassed!\n");
	else
		printf("\nFailed!\n");
	getTime(&t);
	printTime(&s, &t, "Check: ", "\n");

	free(nThread);
	free(refOutput);

	ecl::free(randArray);
	ecl::free(output);

#ifdef _MSC_VER
	system("pause");
#endif

	return 0;
}