#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gmac/opencl.h"

#include "utils.h"

#include "../eclMonteCarloAsianKernel.cl"

#define GROUP_SIZE 256

typedef struct _MonteCalroAttrib
{
	cl_float strikePrice[4];
	cl_float c1[4];
	cl_float c2[4];
	cl_float c3[4];
	cl_float initPrice[4];
	cl_float sigma[4];
	cl_float timeStep[4];
}MonteCarloAttrib;

const cl_float maturity = 1.f;                  /**< maturity */
const cl_float interest = 0.06f;                  /**< Interest rate */
const cl_int noOfTraj = 1024;                    /**< Number of samples */
const cl_float initPrice = 50.f;                /**< Initial price */
const cl_float strikePrice = 55.f;               /**< Stirke price */
cl_int steps = 10;                       /**< Steps for Asian Monte Carlo simution */
cl_int height = 0;
MonteCarloAttrib attributes;
cl_int width = 0;
cl_int noOfSum = 12;

void
lshift128(unsigned int* input, unsigned int shift, unsigned int* output)
{
	unsigned int invshift = 32u - shift;

	output[0] = input[0] << shift;
	output[1] = (input[1] << shift) | (input[0] >> invshift);
	output[2] = (input[2] << shift) | (input[1] >> invshift);
	output[3] = (input[3] << shift) | (input[2] >> invshift);
}

void
rshift128(unsigned int* input, unsigned int shift, unsigned int* output)
{
	unsigned int invshift = 32u - shift;

	output[3]= input[3] >> shift;
	output[2] = (input[2] >> shift) | (input[0] >> invshift);
	output[1] = (input[1] >> shift) | (input[1] >> invshift);
	output[0] = (input[0] >> shift) | (input[2] >> invshift);
}

void 
generateRand(unsigned int* seed, float* gaussianRand1, float* gaussianRand2, unsigned int* nextRand)
{
	unsigned int mulFactor = 4;
	unsigned int temp[8][4];

	unsigned int state1[4] = {seed[0], seed[1], seed[2], seed[3]};
	unsigned int state2[4] = {0u, 0u, 0u, 0u}; 
	unsigned int state3[4] = {0u, 0u, 0u, 0u}; 
	unsigned int state4[4] = {0u, 0u, 0u, 0u}; 
	unsigned int state5[4] = {0u, 0u, 0u, 0u}; 

	unsigned int stateMask = 1812433253u;
	unsigned int thirty = 30u;
	unsigned int mask4[4] = {stateMask, stateMask, stateMask, stateMask};
	unsigned int thirty4[4] = {thirty, thirty, thirty, thirty};
	unsigned int one4[4] = {1u, 1u, 1u, 1u};
	unsigned int two4[4] = {2u, 2u, 2u, 2u};
	unsigned int three4[4] = {3u, 3u, 3u, 3u};
	unsigned int four4[4] = {4u, 4u, 4u, 4u};

	unsigned int r1[4] = {0u, 0u, 0u, 0u}; 
	unsigned int r2[4] = {0u, 0u, 0u, 0u}; 

	unsigned int a[4] = {0u, 0u, 0u, 0u};
	unsigned int b[4] = {0u, 0u, 0u, 0u};

	unsigned int e[4] = {0u, 0u, 0u, 0u}; 
	unsigned int f[4] = {0u, 0u, 0u, 0u};

	unsigned int thirteen  = 13u;
	unsigned int fifteen = 15u;
	unsigned int shift = 8u * 3u;

	unsigned int mask11 = 0xfdff37ffu;
	unsigned int mask12 = 0xef7f3f7du;
	unsigned int mask13 = 0xff777b7du;
	unsigned int mask14 = 0x7ff7fb2fu;

	const float one = 1.0f;
	const float intMax = 4294967296.0f;
	const float PI = 3.14159265358979f;
	const float two = 2.0f;

	float r[4] = {0.0f, 0.0f, 0.0f, 0.0f}; 
	float phi[4] = {0.0f, 0.0f, 0.0f, 0.0f};

	float temp1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	float temp2[4] = {0.0f, 0.0f, 0.0f, 0.0f};

	//Initializing states.
	for(int c = 0; c < 4; ++c)	{
		state2[c] = mask4[c] * (state1[c] ^ (state1[c] >> thirty4[c])) + one4[c];
		state3[c] = mask4[c] * (state2[c] ^ (state2[c] >> thirty4[c])) + two4[c];
		state4[c] = mask4[c] * (state3[c] ^ (state3[c] >> thirty4[c])) + three4[c];
		state5[c] = mask4[c] * (state4[c] ^ (state4[c] >> thirty4[c])) + four4[c];
	}

	unsigned int i = 0;
	for(i = 0; i < mulFactor; ++i) {
		switch(i)
		{
		case 0:
			for(int c = 0; c < 4; ++c) {
				r1[c] = state4[c];
				r2[c] = state5[c];
				a[c] = state1[c];
				b[c] = state3[c];
			}
			break;
		case 1:
			for(int c = 0; c < 4; ++c) {
				r1[c] = r2[c];
				r2[c] = temp[0][c];
				a[c] = state2[c];
				b[c] = state4[c];
			}
			break;
		case 2:
			for(int c = 0; c < 4; ++c) {
				r1[c] = r2[c];
				r2[c] = temp[1][c];
				a[c] = state3[c];
				b[c] = state5[c];
			}
			break;
		case 3:
			for(int c = 0; c < 4; ++c) {
				r1[c] = r2[c];
				r2[c] = temp[2][c];
				a[c] = state4[c];
				b[c] = state1[c];
			}
			break;
		default:
			break;
		}

		lshift128(a, shift, e);
		rshift128(r1, shift, f);

		temp[i][0] = a[0] ^ e[0] ^ ((b[0] >> thirteen) & mask11) ^ f[0] ^ (r2[0] << fifteen);
		temp[i][1] = a[1] ^ e[1] ^ ((b[1] >> thirteen) & mask12) ^ f[1] ^ (r2[1] << fifteen);
		temp[i][2] = a[2] ^ e[2] ^ ((b[2] >> thirteen) & mask13) ^ f[2] ^ (r2[2] << fifteen);
		temp[i][3] = a[3] ^ e[3] ^ ((b[3] >> thirteen) & mask14) ^ f[3] ^ (r2[3] << fifteen);
	}

	for(int c = 0; c < 4; ++c)	{
		temp1[c] = temp[0][c] * one / intMax;
		temp2[c] = temp[1][c] * one / intMax;
	}

	for(int c = 0; c < 4; ++c)	{ 
		// Applying Box Mullar Transformations.
		r[c] = sqrt((-two) * log(temp1[c]));
		phi[c]  = two * PI * temp2[c];
		gaussianRand1[c] = r[c] * cos(phi[c]);
		gaussianRand2[c] = r[c] * sin(phi[c]);

		nextRand[c] = temp[2][c];
	}
}

void 
calOutputs(float strikePrice, float* meanDeriv1, float*  meanDeriv2, float* meanPrice1, float* meanPrice2, float* pathDeriv1, float* pathDeriv2, float* priceVec1, float* priceVec2)
{
	float temp1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	float temp2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	float temp3[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	float temp4[4] = {0.0f, 0.0f, 0.0f, 0.0f};

	float tempDiff1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	float tempDiff2[4] = {0.0f, 0.0f, 0.0f, 0.0f};

	for(int c = 0; c < 4; ++c)	{
		tempDiff1[c] = meanPrice1[c] - strikePrice;
		tempDiff2[c] = meanPrice2[c] - strikePrice;
	}
	if(tempDiff1[0] > 0.0f)
	{
		temp1[0] = 1.0f;
		temp3[0] = tempDiff1[0];
	}
	if(tempDiff1[1] > 0.0f)
	{
		temp1[1] = 1.0f;
		temp3[1] = tempDiff1[1];
	}
	if(tempDiff1[2] > 0.0f)
	{
		temp1[2] = 1.0f;
		temp3[2] = tempDiff1[2];
	}
	if(tempDiff1[3] > 0.0f)
	{
		temp1[3] = 1.0f;
		temp3[3] = tempDiff1[3];
	}

	if(tempDiff2[0] > 0.0f)
	{
		temp2[0] = 1.0f;
		temp4[0] = tempDiff2[0];
	}
	if(tempDiff2[1] > 0.0f)
	{
		temp2[1] = 1.0f;
		temp4[1] = tempDiff2[1];
	}
	if(tempDiff2[2] > 0.0f)
	{
		temp2[2] = 1.0f;
		temp4[2] = tempDiff2[2];
	}
	if(tempDiff2[3] > 0.0f)
	{
		temp2[3] = 1.0f;
		temp4[3] = tempDiff2[3];
	}

	for(int c = 0; c < 4; ++c)	{
		pathDeriv1[c] = meanDeriv1[c] * temp1[c]; 
		pathDeriv2[c] = meanDeriv2[c] * temp2[c]; 
		priceVec1[c] = temp3[c]; 
		priceVec2[c] = temp4[c];
	}
}

void cpuReferenceImpl(cl_float* sigma, cl_float* refPrice, cl_float* refVega, cl_float* priceVals, cl_float* priceDeriv)
{
	float timeStep = maturity / (noOfSum - 1);

	// Initialize random number generator
	srand(1);

	for(int k = 0; k < steps; k++) {
		float c1 = (interest - 0.5f * sigma[k] * sigma[k]) * timeStep;
		float c2 = sigma[k] * sqrt(timeStep);
		float c3 = (interest + 0.5f * sigma[k] * sigma[k]); 

		for(int j = 0; j < (width * height); j++) {
			unsigned int nextRand[4] = {0u, 0u, 0u, 0u};
			for(int c = 0; c < 4; ++c)
				nextRand[c] = (cl_uint)rand();

			float trajPrice1[4] = {initPrice, initPrice, initPrice, initPrice};
			float sumPrice1[4] = {initPrice, initPrice, initPrice, initPrice};
			float sumDeriv1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
			float meanPrice1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
			float meanDeriv1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
			float price1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
			float pathDeriv1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

			float trajPrice2[4] = {initPrice, initPrice, initPrice, initPrice};
			float sumPrice2[4] = {initPrice, initPrice, initPrice, initPrice};
			float sumDeriv2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
			float meanPrice2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
			float meanDeriv2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
			float price2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
			float pathDeriv2[4] = {0.0f, 0.0f, 0.0f, 0.0f};

			//Run the Monte Carlo simulation a total of Num_Sum - 1 times
			for(int i = 1; i < noOfSum; i++) {
				unsigned int tempRand[4] =  {0u, 0u, 0u, 0u};
				for(int c = 0; c < 4; ++c)
					tempRand[c] = nextRand[c];

				float gaussian1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
				float gaussian2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
				generateRand(tempRand, gaussian1, gaussian2, nextRand);

				//Calculate the trajectory price and sum price for all trajectories
				for(int c = 0; c < 4; ++c) {
					trajPrice1[c] = trajPrice1[c] * exp(c1 + c2 * gaussian1[c]);
					trajPrice2[c] = trajPrice2[c] * exp(c1 + c2 * gaussian2[c]);

					sumPrice1[c] = sumPrice1[c] + trajPrice1[c];
					sumPrice2[c] = sumPrice2[c] + trajPrice2[c];

					float temp = c3 * timeStep * i;

					// Calculate the derivative price for all trajectories
					sumDeriv1[c] = sumDeriv1[c] + trajPrice1[c] 
					* ((log(trajPrice1[c] / initPrice) - temp) / sigma[k]);

					sumDeriv2[c] = sumDeriv2[c] + trajPrice2[c] 
					* ((log(trajPrice2[c] / initPrice) - temp) / sigma[k]);						
				}
			}

			//Calculate the average price and average derivative?of each simulated path
			for(int c = 0; c < 4; ++c) {
				meanPrice1[c] = sumPrice1[c] / noOfSum;
				meanPrice2[c] = sumPrice2[c] / noOfSum;
				meanDeriv1[c] = sumDeriv1[c] / noOfSum;
				meanDeriv2[c] = sumDeriv2[c] / noOfSum;
			}

			calOutputs(strikePrice, meanDeriv1, meanDeriv2, meanPrice1, meanPrice2, 
				pathDeriv1, pathDeriv2, price1, price2);

			for(int c = 0; c < 4; ++c)	{
				priceVals[j * 8 + c] = price1[c];
				priceVals[j * 8 + 1 * 4 + c] = price2[c];
				priceDeriv[j * 8 + c] = pathDeriv1[c];
				priceDeriv[j * 8 + 1 * 4 + c] = pathDeriv2[c];
			}
		}

		/* Replace Following "for" loop with reduction kernel */
		for(int i = 0; i < noOfTraj * noOfTraj; i++)	{
			refPrice[k] += priceVals[i];
			refVega[k] += priceDeriv[i];
		}

		refPrice[k] /= (noOfTraj * noOfTraj);
		refVega[k] /= (noOfTraj * noOfTraj);

		refPrice[k] = exp(-interest * maturity) * refPrice[k];
		refVega[k] = exp(-interest * maturity) * refVega[k];
	}
}

int main(int argc, char *argv[])
{
	gmactime_t s, t, S , T;

	size_t blockSizeX = GROUP_SIZE;                  /**< Group-size in x-direction */ 
	size_t blockSizeY = 1;                                      /**< Group-size in y-direction */

	cl_float *sigma = NULL;                    /**< Array of sigma values */
	cl_float *price = NULL;                    /**< Array of price values */
	cl_float *vega = NULL;                     /**< Array of vega values */
	cl_float *refPrice = NULL;                 /**< Array of reference price values */
	cl_float *refVega = NULL;                  /**< Array of reference vega values */
	cl_uint *randNum = NULL;                   /**< Array of random numbers */
	cl_float *priceVals = NULL;                /**< Array of price values for given samples */
	cl_float *priceDeriv = NULL;               /**< Array of price derivative values for given samples */
	ecl_error error_code = eclSuccess;

	steps = (steps < 4) ? 4 : steps;
	steps = (steps / 2) * 2;

	int i = 0;
	const cl_float finalValue = 0.8f;
	const cl_float stepValue = finalValue / (cl_float)steps;

	error_code = eclCompileSource(code);
	assert(error_code == eclSuccess);

	/* Set samples and exercize points */
	width = noOfTraj / 4;
	height = noOfTraj / 2;

	getTime(&s);
	// Alloc
	error_code = eclMalloc((void **)&sigma, steps * sizeof(cl_float));
	assert(error_code == eclSuccess);
	error_code = eclMalloc((void **)&price, steps * sizeof(cl_float));
	assert(error_code == eclSuccess);
	error_code = eclMalloc((void **)&vega, steps * sizeof(cl_float));
	assert(error_code == eclSuccess);

	refPrice = (cl_float*) malloc(steps * sizeof(cl_float));
	assert(refPrice != NULL);
	refVega = (cl_float*) malloc(steps * sizeof(cl_float));
	assert(refVega != NULL);

	error_code = eclMalloc((void **)&randNum, width * height * sizeof(cl_uint4));
	assert(error_code == eclSuccess);
	error_code = eclMalloc((void **)&priceVals, width * height * 2 * sizeof(cl_float4));
	assert(error_code == eclSuccess);
	error_code = eclMalloc((void **)&priceDeriv, width * height * 2 * sizeof(cl_float4));
	assert(error_code == eclSuccess);

	getTime(&t);
	printTime(&s, &t, "Alloc: ", "\n");

	getTime(&S);
	getTime(&s);
	/* random initialisation of input */
	sigma[0] = 0.01f;
	for(i = 1; i < steps; i++)
		sigma[i] = sigma[i - 1] + stepValue;
	memset((void*)price, 0, steps * sizeof(cl_float));
	memset((void*)vega, 0, steps * sizeof(cl_float));
	memset((void*)refPrice, 0, steps * sizeof(cl_float));
	memset((void*)refVega, 0, steps * sizeof(cl_float));
	memset((void*)priceVals, 0, width * height * 2 * sizeof(cl_float4));
	memset((void*)priceDeriv, 0, width * height * 2 * sizeof(cl_float4));

	getTime(&t);
	printTime(&s, &t, "Init: ", "\n");

	getTime(&s);
	// Print the input data
	printf("\nSigma:\n");
	for(cl_int i = 0; i < steps; i++)
		printf("%f ", sigma[i]);
	getTime(&t);
	printTime(&s, &t, "\nPrint: ", "\n");

	getTime(&s);
	size_t globalThreads[] = {width, height};
	size_t localThreads[] = {blockSizeX, blockSizeY};
	ecl_kernel kernel;	
	error_code = eclGetKernel("calPriceVega", &kernel);
	assert(error_code == eclSuccess);
	error_code = eclSetKernelArg(kernel, 2, sizeof(width), &width);
	assert(error_code == eclSuccess);
	error_code = eclSetKernelArg(kernel, 1, sizeof(noOfSum), &noOfSum);
	assert(error_code == eclSuccess);
	error_code = eclSetKernelArgPtr(kernel, 3, randNum);
	assert(error_code == eclSuccess);
	error_code = eclSetKernelArgPtr(kernel, 4,  priceVals);
	assert(error_code == eclSuccess);
	error_code = eclSetKernelArgPtr(kernel, 5, priceDeriv);
	assert(error_code == eclSuccess);

	cl_float timeStep = maturity / (noOfSum - 1);
	// Initialize random number generator
	srand(1);
	for(int k = 0; k < steps; k++) {
		for(int j = 0; j < (width * height * 4); j++)
			randNum[j] = (cl_uint)rand();
		cl_float c1 = (interest - 0.5f * sigma[k] * sigma[k]) * timeStep;
		cl_float c2 = sigma[k] * sqrt(timeStep);
		cl_float c3 = (interest + 0.5f * sigma[k] * sigma[k]);

		attributes.c1[0] = c1;
		attributes.c1[1] = c1;
		attributes.c1[2] = c1;
		attributes.c1[3] = c1;
		attributes.c2[0] = c2;
		attributes.c2[1] = c2;
		attributes.c2[2] = c2;
		attributes.c2[3] = c2;
		attributes.c3[0] = c3;
		attributes.c3[1] = c3;
		attributes.c3[2] = c3;
		attributes.c3[3] = c3;
		attributes.initPrice[0] = initPrice;
		attributes.initPrice[1] = initPrice;
		attributes.initPrice[2] = initPrice;
		attributes.initPrice[3] = initPrice;
		attributes.strikePrice[0] = strikePrice;
		attributes.strikePrice[1] = strikePrice;
		attributes.strikePrice[2] = strikePrice;
		attributes.strikePrice[3] = strikePrice;
		attributes.sigma[0] = sigma[k];
		attributes.sigma[1] = sigma[k];
		attributes.sigma[2] = sigma[k];
		attributes.sigma[3] = sigma[k];
		attributes.timeStep[0] = timeStep;
		attributes.timeStep[1] = timeStep;
		attributes.timeStep[2] = timeStep;
		attributes.timeStep[3] = timeStep;

		// Set appropriate arguments to the kernel 
		error_code = eclSetKernelArg(kernel, 0, sizeof(attributes), &attributes);
		assert(error_code == eclSuccess);
		error_code = eclCallNDRange(kernel, 2, NULL, globalThreads, localThreads);
		assert(error_code == eclSuccess);

		/* Replace Following "for" loop with reduction kernel */
		for(int i = 0; i < noOfTraj * noOfTraj; i++) {
			price[k] += priceVals[i];
			vega[k] += priceDeriv[i];
		}
		price[k] /= (noOfTraj * noOfTraj);
		vega[k] /= (noOfTraj * noOfTraj);

		price[k] = exp(-interest * maturity) * price[k];
		vega[k] = exp(-interest * maturity) * vega[k];
	}
	getTime(&t);
	printTime(&s, &t, "Run: ", "\n");

	printf("\nprice: \n");
	for(cl_int i = 0; i < steps; i++)
		printf("%f ", price[i]);
	printf("\nvega:\n");
	for(cl_int i = 0; i < steps; i++)
		printf("%f ", vega[i]);

	getTime(&s);
	cpuReferenceImpl(sigma, refPrice, refVega, priceVals, priceDeriv);
	/* compare the results and see if they match */
	for(int i = 0; i < steps; ++i) {
		if(fabs(price[i] - refPrice[i]) > 0.2f)
		{
			printf( "\nFailed\n");
		}
		if(fabs(vega[i] - refVega[i]) > 0.2f)
		{
			printf( "\nFailed\n");
		}
	}
	printf( "\nPassed\n");

	getTime(&t);
	printTime(&s, &t, "Check: ", "\n");
	getTime(&T);
	printTime(&S, &T, "Total: ", "\n");

	error_code = eclReleaseKernel(kernel);
	assert(error_code == eclSuccess);

	getTime(&s);
	free(refPrice);
	refPrice = NULL;
	free(refVega);
	refVega = NULL;
	error_code = eclFree(sigma);
	assert(error_code == eclSuccess);
	error_code = eclFree(price);
	assert(error_code == eclSuccess);
	error_code = eclFree(vega);
	assert(error_code == eclSuccess);
	error_code = eclFree(randNum);
	assert(error_code == eclSuccess);
	error_code = eclFree(priceVals);
	assert(error_code == eclSuccess);
	error_code = eclFree(priceDeriv);
	assert(error_code == eclSuccess);
	getTime(&t);
	printTime(&s, &t, "Free: ", "\n");

	return 0;
}
