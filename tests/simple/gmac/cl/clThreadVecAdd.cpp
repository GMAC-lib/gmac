/* two thread: shared platform, shared device, shared context, shared command_queue
 * BUT: seperate data collection
 *
 *
 */
/******************************************************************************************************************
 *						RESULT
 *	1. Threads with separate platforms, devices, contexts, command_queues: works well (actually No shared data, No syn. operations)
 *	2. Threads with shared platform, but separate devices, contexts, command_queues: works well (actually No shared data, No syn. operations)
 *	3. Threads with shared platform, devices, but separate contexts, command_queues: works well (actually No shared data, No syn. operations)
 *	4. Threads with shared platform, devices, but separate contexts, command_queues: works well (actually No shared data, No syn. operations)
 *
 *
 *
 */

//#ifdef GMAC_ENABLE
#if 1
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cassert>

#include "gmac/cl.h"
#include "utils.h"
#include "debug.h"

#ifdef _MSC_VER
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif

static const char *vecSizeStr = "GMAC_VECSIZE";
static const unsigned vecSizeDefault = 16 * 1024 * 1024;
static unsigned vecSize = vecSizeDefault;

static const char *kernel_source = "\
                          __kernel void vecAdd(__global float *c, __global const float *a, __global const float *b, unsigned size)\
                          {                                  \
                              unsigned i = get_global_id(0); \
                              if(i >= size) return;          \
                                                             \
                              c[i] = a[i] + b[i];            \
                          }                                  \
                          ";

typedef struct __OpenCLEnv
{
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue command_queue;
} OpenCLEnv;

static OpenCLEnv openCLEnv = {NULL,NULL,NULL,NULL};

#define SHARED_CONTEXT
#define SHARED_COMMAND_QUEUE

#ifdef SHARED_CONTEXT
cl_program program;
#endif

void* Thread(void *_name)
{
    const char *name = (const char *) _name;
	cl_int error_code;
	cl_context context;
	cl_command_queue command_queue;
    cl_kernel kernel;
	float *a, *b, *c;
	gmactime_t s, t;
    char buffer[256];
#ifndef SHARED_CONTEXT
    cl_platform_id platform;
    cl_device_id device;
#endif


#ifndef SHARED_CONTEXT
	platform = openCLEnv.platform;
	device = openCLEnv.device;

	context = clCreateContext(0, 1, &device, NULL, NULL, &error_code);
	assert(error_code == CL_SUCCESS);
#else
	context = openCLEnv.context;
#endif

#ifndef SHARED_COMMAND_QUEUE
	command_queue = clCreateCommandQueue(context, device, 0, &error_code);
	assert(error_code == CL_SUCCESS);
#else
	command_queue = openCLEnv.command_queue;
#endif

	fprintf(stdout, "%s: Vector: %f\n", name, 1.0 * vecSize / 1024 / 1024);

#ifndef SHARED_CONTEXT
    cl_program program;
	program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &error_code);
	assert(error_code == CL_SUCCESS);
	error_code = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	assert(error_code == CL_SUCCESS);
#endif

	kernel = clCreateKernel(program, "vecAdd", &error_code);
	assert(error_code == CL_SUCCESS);

	getTime(&s);

	error_code = clMalloc(command_queue, (void **) &a, vecSize * sizeof(float));
	assert(error_code == CL_SUCCESS);
	error_code = clMalloc(command_queue, (void **) &b, vecSize * sizeof(float));
	assert(error_code == CL_SUCCESS);
	error_code = clMalloc(command_queue, (void **) &c, vecSize * sizeof(float));
	assert(error_code == CL_SUCCESS);
	getTime(&t);

    SNPRINTF(buffer, 256, "%s: Alloc: ", name);
	printTime(&s, &t, buffer, "\n");

	float sum = 0.f;
	getTime(&s);
	valueInit(a, 1.1f, vecSize);
	valueInit(b, 1.f, vecSize);
	getTime(&t);

    SNPRINTF(buffer, 256, "%s: Init: ", name);
	printTime(&s, &t, buffer, "\n");

	for(unsigned i = 0; i < vecSize; i++) {
		sum += a[i] + b[i];
	}

	getTime(&s);
	size_t global_size = vecSize;

	cl_mem c_device = clGetBuffer(context, c);
	error_code = clSetKernelArg(kernel, 0, sizeof(cl_mem), &c_device);
	assert(error_code == CL_SUCCESS);
	cl_mem a_device = clGetBuffer(context, a);
	error_code = clSetKernelArg(kernel, 1, sizeof(cl_mem), &a_device);
	assert(error_code == CL_SUCCESS);
	cl_mem b_device = clGetBuffer(context, b);
	error_code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &b_device);
	assert(error_code == CL_SUCCESS);
	error_code = clSetKernelArg(kernel, 3, sizeof(vecSize), &vecSize);
	assert(error_code == CL_SUCCESS);

	error_code  = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
	assert(error_code == CL_SUCCESS);
	error_code = clFinish(command_queue);
	assert(error_code == CL_SUCCESS);

	fprintf(stdout, "%s: C[0]: %f\n", name, c[0]);

	getTime(&t);
    SNPRINTF(buffer, 256, "%s: Run: ", name);
	printTime(&s, &t, buffer, "\n");

	getTime(&s);
	float error = 0.f;
	float check = 0.f;
	for(unsigned i = 0; i < vecSize; i++) {
		error += c[i] - (a[i] + b[i]);
		check += c[i];
	}
	fprintf(stdout, "%s: Error: %f\n", name, error);

	if (sum != check) {
		fprintf(stdout, "%s: Sum: %f vs %f\n", name, sum, check);
		abort();
	}

	fprintf(stdout, "%s: clFree\n", name);
	error_code = clFree(command_queue, a);
	assert(error_code == CL_SUCCESS);
	fprintf(stdout, "%s: clFree out 1\n", name);
	error_code = clFree(command_queue, b);
	assert(error_code == CL_SUCCESS);
	fprintf(stdout, "%s: clFree out 2\n", name);
	error_code = clFree(command_queue, c);
	assert(error_code == CL_SUCCESS);
	fprintf(stdout, "%s: clFree out 3 OVER\n", name);

	/* Release OpenCL resources */
	error_code = clReleaseKernel(kernel);
	assert(error_code == CL_SUCCESS);
	error_code = clReleaseProgram(program);
	assert(error_code == CL_SUCCESS);
	error_code = clReleaseCommandQueue(command_queue);
	assert(error_code == CL_SUCCESS);
	error_code = clReleaseContext(context);
	assert(error_code == CL_SUCCESS);

    return NULL;
}

int main(int argc, char *argv[])
{
    cl_int error_code;

    /*****************************************************************************************/
    /*		Handle user parameters                                                           */
    /*****************************************************************************************/
	setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);              ///////

    /*****************************************************************************************/
    /*		Initialize the environment								 	  			     	 */
    /*****************************************************************************************/
    error_code = clGetPlatformIDs(1, &(openCLEnv.platform), NULL);
    assert(error_code == CL_SUCCESS);
    error_code = clGetDeviceIDs(openCLEnv.platform, CL_DEVICE_TYPE_GPU, 1, &(openCLEnv.device), NULL);
    assert(error_code == CL_SUCCESS);
    openCLEnv.context = clCreateContext(0, 1, &(openCLEnv.device), NULL, NULL, &error_code);
    assert(error_code == CL_SUCCESS);
    openCLEnv.command_queue = clCreateCommandQueue(openCLEnv.context, openCLEnv.device, 0, &error_code);
    assert(error_code == CL_SUCCESS);

#ifdef SHARED_CONTEXT
	program = clCreateProgramWithSource(openCLEnv.context, 1, &kernel_source, NULL, &error_code);
	assert(error_code == CL_SUCCESS);
	error_code = clBuildProgram(program, 1, &openCLEnv.device, NULL, NULL, NULL);
	assert(error_code == CL_SUCCESS);
#endif

    /************************************************************************/
    /* Invoke the threads                                                   */
    /************************************************************************/
    char name1[] = "Thread A";
    char name2[] = "Thread B";
    char name3[] = "Thread C";
    char name4[] = "Thread D";
    char name5[] = "Thread E";

    thread_t thread1 = thread_create(thread_routine(Thread), name1);
    thread_t thread2 = thread_create(thread_routine(Thread), name2);
    thread_t thread3 = thread_create(thread_routine(Thread), name3);
    thread_t thread4 = thread_create(thread_routine(Thread), name4);
    thread_t thread5 = thread_create(thread_routine(Thread), name5);

    /************************************************************************/
    /* Waiting for ending of the thread                                     */
    /************************************************************************/
    thread_wait(thread1);
    thread_wait(thread2);
    thread_wait(thread3);	
    thread_wait(thread4);	
    thread_wait(thread5);	
#ifdef _MSC_VER
    system("pause");
#endif

    return 0;
}
#endif
