#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "gmac/opencl"
#include "utils.h"

#ifdef _MSC_VER
#define SLEEP Sleep
#define SNPRINTF sprintf_s
#else
#include <unistd.h>
#define SLEEP sleep
#define SNPRINTF snprintf
#endif

static unsigned vecSize = 16 * 1024 * 1024;

const char *nIterStr = "GMAC_NITER";
const char *vecSizeStr = "GMAC_VECSIZE";

static const char *kernel = "\
							__kernel void vecAdd(__global float *c, __global float *a, __global float *b)\
							{\
							unsigned i = get_global_id(0);\
							\
							c[i] = a[i] + b[i];\
							}\
							";
static const char *kernel_A = "\
							  __kernel void vecAdd_A(__global float *c, __global float *a, __global float *b)\
							  {\
							  unsigned i = get_global_id(0);\
							  \
							  c[i] = a[i] + b[i];\
							  }\
							  ";

static float *resultA;
static THREAD_T threadIdA;
static THREAD_T threadIdB;
static THREAD_T threadIdC;
static THREAD_T threadIdD;
static THREAD_T threadIdE;

static void *ThreadBody_First(void *)
{
	float *a, *b;

	fprintf(stdout, "Thread A: Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

	// Alloc & init input data
	assert(ecl::malloc((void **)&a, vecSize * sizeof(float)) == eclSuccess);
	assert(ecl::malloc((void **)&b, vecSize * sizeof(float)) == eclSuccess);
	// Alloc output data
	assert(ecl::malloc((void **)&resultA, vecSize * sizeof(float)) == eclSuccess);

	for(unsigned i = 0; i < vecSize; i++) {
		a[i] = 1.f;
		b[i] = 1.f;
	}

	// Call the kernel
	ecl::error ret;
	ecl::config globalSize(vecSize);
	ecl::config localSize(1);
	ecl::config globalWorkOffset(0);
	ecl::kernel kernel("vecAdd", ret);
	assert(ret == eclSuccess);

	ret = kernel.setArg(0, resultA);
	assert(ret == eclSuccess);
	ret = kernel.setArg(1, a);
	assert(ret == eclSuccess);
	ret = kernel.setArg(2, b);
	assert(ret == eclSuccess);
	ret = kernel.callNDRange(globalSize, localSize, globalWorkOffset);
	assert(ret == eclSuccess);

	// Check the result in the CPU
	float error = 0.f;
	for(unsigned i = 0; i < vecSize; i++) {
		error += (resultA[i] - (a[i] + b[i]));
	}
	fprintf(stderr, "Thread A: Error: %f\n", error);

	ecl::free(a);
	ecl::free(b);

	return NULL;
}

struct print_params {
	const char *name;
	float *input;
};

static void *ThreadBody_Print(void *_params)
{
	print_params *params = (print_params *) _params;
	char buffer[256];

	SNPRINTF(buffer, 256, "%s:", params->name);
	for(unsigned i = 0; i < 10; i++) {
		SNPRINTF(buffer, 256, "%s %f ", buffer, params->input[i]);
	}
	fprintf(stdout, "%s\n", buffer);

	return NULL;
}

static void *ThreadBody_Second(void *input)
{
	float* input_E = *(float**)input;
	float *resultB, *temp, *temp_2;
	char buffer[256];

	SNPRINTF(buffer, 256, "Thread E:");
	for(unsigned i = 0; i < 10; i++) {
		SNPRINTF(buffer, 256, "%s %f ", buffer, input_E[i]);
	}
	fprintf(stdout, "%s\n", buffer);

	assert(ecl::compileSource(kernel_A) == eclSuccess);

	fprintf(stdout, "Thread E: Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

	assert(ecl::malloc((void **)&temp, vecSize * sizeof(float)) == eclSuccess);
	assert(ecl::malloc((void **)&resultB, vecSize * sizeof(float)) == eclSuccess);
	assert(ecl::malloc((void **)&temp_2, vecSize * sizeof(float)) == eclSuccess);

	ecl::memcpy(temp_2, resultA, vecSize * sizeof(float));

	for(unsigned i = 0; i < vecSize; i++) {
		temp[i] = 1.f; 
	}

	ecl::error ret;
	ecl::config globalSize(vecSize);
	ecl::config localSize(1);
	ecl::config globalWorkOffset(0);
	ecl::kernel kernel("vecAdd_A", ret);
	assert(ret == eclSuccess);

	ret = kernel.setArg(0, resultB);
	assert(ret == eclSuccess);
	ret = kernel.setArg(1, temp);
	assert(ret == eclSuccess);
	//ret = kernel.setArg(2, resultA);
	//assert(ret == eclSuccess);
	ret = kernel.setArg(2, temp_2);
	assert(ret == eclSuccess);
	ret = kernel.callNDRange(globalSize, localSize, globalWorkOffset);
	assert(ret == eclSuccess);

	// Check the result in the CPU
	float error = 0.f;
	for(unsigned i = 0; i < vecSize; i++) {
		error += resultB[i] - (temp[i] + temp_2[i]);	
	}
	fprintf(stdout, "resultB: %f \n", resultB[0]);
	fprintf(stdout, "Thread E: Error: %f\n", error);

	ecl::free(temp);
	ecl::free(resultB);
	ecl::free(temp_2);

	return NULL;
}

void *addVector(void *ptr)
{
	return NULL;
}

int main(int argc, char *argv[])
{
	/* Thread B C D E wait for thread A to end
	* Thread B C D read the computation result from thread A
	* Thread E use the result of thread A to calculate
	*/
	assert(ecl::compileSource(kernel) == eclSuccess);

	threadIdA = thread_create(thread_routine(ThreadBody_First),NULL);
	SLEEP(10);

	thread_wait(threadIdA);

	print_params p1, p2, p3;

	p1.name  = "Thread B";
	p1.input = resultA;

	p2.name  = "Thread C";
	p2.input = resultA;

	p3.name  = "Thread D";
	p3.input = resultA;

	threadIdB = thread_create(thread_routine(ThreadBody_Print),&p1);
	threadIdC = thread_create(thread_routine(ThreadBody_Print),&p2);
	threadIdD = thread_create(thread_routine(ThreadBody_Print),&p3);
	threadIdE = thread_create(thread_routine(ThreadBody_Second),&resultA);
	thread_wait(threadIdB);
	thread_wait(threadIdC);
	thread_wait(threadIdD);
	thread_wait(threadIdE);

	fprintf(stdout, "main: ");
	for(unsigned i = 0; i < 10; i++) {
		fprintf(stdout, "%f ", resultA[i]);
	}
	fprintf(stdout, "\n");

	ecl::free(resultA);

#ifdef _MSC_VER
	system("pause");
#endif

	return 0;
}
