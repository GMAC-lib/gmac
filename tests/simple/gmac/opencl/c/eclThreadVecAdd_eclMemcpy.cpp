#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "gmac/opencl.h"
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
 	assert(eclMalloc((void **)&a, vecSize * sizeof(float)) == eclSuccess);
 	assert(eclMalloc((void **)&b, vecSize * sizeof(float)) == eclSuccess);
 	// Alloc output data
 	assert(eclMalloc((void **)&resultA, vecSize * sizeof(float)) == eclSuccess);
	//assert(eclMalloc(input, vecSize * sizeof(float)) == eclSuccess);


 	for(unsigned i = 0; i < vecSize; i++) {
 		a[i] = 1.f;
 		b[i] = 1.f;
 	}

	// Call the kernel
	size_t globalSize = vecSize;
	ecl_kernel kernel;
	assert(eclGetKernel("vecAdd", &kernel) == eclSuccess);
	assert(eclSetKernelArgPtr(kernel, 0, resultA) == eclSuccess);
	assert(eclSetKernelArgPtr(kernel, 1, a) == eclSuccess);
	assert(eclSetKernelArgPtr(kernel, 2, b) == eclSuccess);
	assert(eclCallNDRange(kernel, 1, NULL, &globalSize, NULL) == eclSuccess);

	// Check the result in the CPU
	float error = 0.f;
	for(unsigned i = 0; i < vecSize; i++) {
		error += (resultA[i] - (a[i] + b[i]));
	}
	fprintf(stderr, "Thread A: Error: %f\n", error);

	eclReleaseKernel(kernel);

 	eclFree(a);
	eclFree(b);
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

 	assert(eclCompileSource(kernel_A) == eclSuccess);
 
 	fprintf(stdout, "Thread E: Vector: %f\n", 1.0 * vecSize / 1024 / 1024);
 
  	assert(eclMalloc((void **)&temp, vecSize * sizeof(float)) == eclSuccess);
 	assert(eclMalloc((void **)&resultB, vecSize * sizeof(float)) == eclSuccess);
	assert(eclMalloc((void **)&temp_2, vecSize * sizeof(float)) == eclSuccess);

	eclMemcpy(temp_2, resultA, vecSize * sizeof(float));
 	
	for(unsigned i = 0; i < vecSize; i++) {
 		temp[i] = 1.f; 
 	}
	size_t globalSize = vecSize;
	ecl_kernel kernel;	

	assert(eclGetKernel("vecAdd_A", &kernel) == eclSuccess);
	assert(eclSetKernelArgPtr(kernel, 0, resultB) == eclSuccess);
	assert(eclSetKernelArgPtr(kernel, 1, temp) == eclSuccess);
	//assert(eclSetKernelArgPtr(kernel, 2, resultA) == eclSuccess);
	assert(eclSetKernelArgPtr(kernel, 2, temp_2) == eclSuccess);
	assert(eclCallNDRange(kernel, 1, NULL, &globalSize, NULL) == eclSuccess);

 	// Check the result in the CPU
	float error = 0.f;
	for(unsigned i = 0; i < vecSize; i++) {
		error += resultB[i] - (temp[i] + temp_2[i]);	
	}
	fprintf(stdout, "resultB: %f \n", resultB[0]);
	fprintf(stdout, "Thread E: Error: %f\n", error);

	eclReleaseKernel(kernel);

	eclFree(temp);
	eclFree(resultB);
	eclFree(temp_2);

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
	assert(eclCompileSource(kernel) == eclSuccess);

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

	eclFree(resultA);

#ifdef _MSC_VER
	system("pause");
#endif

    return 0;
}
