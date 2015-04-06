#if defined(__APPLE__)
#   include <OpenCL/cl.h>
#else 
#   include <CL/cl.h>
#endif

#include "config/config.h"

#include "api/opencl/lite/Process.h"
#include "include/gmac/cl.h"
#include "libs/common.h"
#include "memory/Handler.h"
#include "memory/Manager.h"
#include "memory/allocator/Slab.h"
#include "util/loader.h"
#include "util/Logger.h"
#include "util/Parameter.h"

#if defined(__GNUC__)
#define RETURN_ADDRESS __builtin_return_address(0)
#elif defined(_MSC_VER)
extern "C" void * _ReturnAddress(void);
#pragma intrinsic(_ReturnAddress)
#define RETURN_ADDRESS _ReturnAddress()
static long getpagesize (void) {
    static long pagesize = 0;
    if(pagesize == 0) {
        SYSTEM_INFO systemInfo;
        GetSystemInfo(&systemInfo);
        pagesize = systemInfo.dwPageSize;
    }
    return pagesize;
}
#endif

#if defined(__APPLE__)
#define CL_CALLBACK 
#endif

static __impl::opencl::lite::Process *Process_ = NULL;
static gmac::memory::Manager *Manager_ = NULL;

STD_SYM(cl_context, __opencl_clCreateContext,
        const cl_context_properties *,
        cl_uint,
        const cl_device_id *,
        void (CL_CALLBACK *)(const char *, const void *, size_t, void *),
        void *,
        cl_int *);

STD_SYM(cl_context, __opencl_clCreateContextFromType,
        const cl_context_properties *,
        cl_device_type,
        void (CL_CALLBACK *)(const char *, const void *, size_t, void *),
        void *,
        cl_int *);

STD_SYM(cl_int, __opencl_clRetainContext, cl_context);

STD_SYM(cl_int, __opencl_clReleaseContext, cl_context);

STD_SYM(cl_command_queue, __opencl_clCreateCommandQueue,
        cl_context,
        cl_device_id,
        cl_command_queue_properties,
        cl_int *);

STD_SYM(cl_int, __opencl_clRetainCommandQueue, cl_command_queue);

STD_SYM(cl_int, __opencl_clReleaseCommandQueue, cl_command_queue);

STD_SYM(cl_int, __opencl_clEnqueueNDRangeKernel,
        cl_command_queue,
        cl_kernel,
        cl_uint,
        const size_t *,
        const size_t *,
        const size_t *,
        cl_uint,
        const cl_event *,
        cl_event *);

STD_SYM(cl_int, __opencl_clEnqueueTask,
        cl_command_queue,
        cl_kernel,
        cl_uint,
        const cl_event *,
        cl_event *);

STD_SYM(cl_int, __opencl_clEnqueueNativeKernel,
        cl_command_queue,
        void (*)(void *),
        void *,
        size_t,
        cl_uint,
        const cl_mem *,
        const void **,
        cl_uint,
        const cl_event *,
        cl_event *);

STD_SYM(cl_int, __opencl_clFinish, cl_command_queue);

using __impl::opencl::lite::Mode;

#ifdef __cplusplus
extern "C" {
#endif

static void openclInit();

cl_context STD_SYMBOL(clCreateContext)(
        const cl_context_properties *properties,
        cl_uint num_devices,
        const cl_device_id *devices,
        void (CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
        void *user_data,
        cl_int *errcode_ret)
{
    if(__opencl_clCreateContext == NULL) openclInit();
    cl_context ret = __opencl_clCreateContext(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
    if(num_devices != 1 || *errcode_ret != CL_SUCCESS) return ret;

    enterGmac();
    Process_->createMode(ret, num_devices, devices);
    exitGmac();

    return ret;
}

cl_context STD_SYMBOL(clCreateContextFromType)(
        const cl_context_properties *properties,
        cl_device_type device_type,
        void (CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
        void *user_data,
        cl_int *errcode_ret)
{
    if(__opencl_clCreateContext == NULL) openclInit();
    cl_context ret = __opencl_clCreateContextFromType(properties, device_type, pfn_notify, user_data, errcode_ret);
    if(*errcode_ret != CL_SUCCESS) return ret;

#if defined(__APPLE__)
    cl_uint num_devices = 1;
    cl_int err = CL_SUCCESS;
#else
    cl_uint num_devices;
    cl_int err = clGetContextInfo(ret, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &num_devices, NULL);
    if(err != CL_SUCCESS || num_devices != 1) return ret;
#endif

    cl_device_id device;
    err = clGetContextInfo(ret, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, NULL);
    if(err != CL_SUCCESS) return ret;

    enterGmac();
    Process_->createMode(ret, num_devices, &device);
    exitGmac();

    return ret;
}

cl_int STD_SYMBOL(clRetainContext)(cl_context context)
{
    if(__opencl_clRetainContext == NULL) openclInit();
    cl_int ret = __opencl_clRetainContext(context);
    if(inGmac() || ret != CL_SUCCESS) return ret;
    enterGmac();
    Process_->getMode(context);
    exitGmac();
    return ret;
}

cl_int STD_SYMBOL(clReleaseContext)(cl_context context)
{
    if(__opencl_clReleaseContext == NULL) openclInit();
    cl_int ret = __opencl_clReleaseContext(context);
    if(inGmac() || ret != CL_SUCCESS) return ret;
    enterGmac();
    Mode *mode = Process_->getMode(context);
    if(mode != NULL) {
        mode->decRef();
        // We decrease the usage count twice to effectively release the mode
        mode->decRef();
    }
    exitGmac();
    return ret;
}

cl_command_queue STD_SYMBOL(clCreateCommandQueue)(
        cl_context context,
        cl_device_id device,
        cl_command_queue_properties properties,
        cl_int *errcode_ret)
{
    if(__opencl_clCreateCommandQueue == NULL) openclInit();
    cl_command_queue ret = __opencl_clCreateCommandQueue(context, device, properties,  errcode_ret);
    if(inGmac() || *errcode_ret != CL_SUCCESS) return ret;
    enterGmac();
    Mode *mode = Process_->getMode(context);
    if(mode == NULL) return ret;
    mode->addQueue(ret);
    mode->decRef();
    exitGmac();
    return ret;
}

cl_int STD_SYMBOL(clRetainCommandQueue)(cl_command_queue command_queue)
{
    if(__opencl_clRetainCommandQueue == NULL) openclInit();
    cl_int ret = __opencl_clRetainCommandQueue(command_queue);
    if(inGmac() || ret != CL_SUCCESS) return ret;
    return ret;
}

cl_int STD_SYMBOL(clReleaseCommandQueue)(cl_command_queue command_queue)
{
    if(__opencl_clReleaseCommandQueue == NULL) openclInit();
    cl_context context;
    cl_int ret = clGetCommandQueueInfo(command_queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, NULL);
    if(ret != CL_SUCCESS) return ret;

    cl_uint count = 0;
    ret = clGetCommandQueueInfo(command_queue, CL_QUEUE_REFERENCE_COUNT, sizeof(cl_uint), &count, NULL);
    if(ret != CL_SUCCESS) return ret;

    ret = __opencl_clReleaseCommandQueue(command_queue);
    if(inGmac() || ret != CL_SUCCESS || count > 1) return ret;
    enterGmac();
    Mode *mode = Process_->getMode(context);
    if(mode == NULL) return ret;
    mode->removeQueue(command_queue);
    mode->decRef();
    exitGmac();
    return ret;
}

#if 0
static void acquireMemoryObjects(cl_event event, cl_int status, void *user_data)
{
    Mode *mode = NULL;
    cl_context context;
    cl_command_queue queue;
    cl_int ret = CL_SUCCESS;
    ret = clGetEventInfo(event, CL_EVENT_CONTEXT, sizeof(cl_context), &context, NULL);
    if(ret != CL_SUCCESS) goto do_exit;
    ret = clGetEventInfo(event, CL_EVENT_COMMAND_QUEUE, sizeof(cl_command_queue), &queue, NULL);
    if(ret != CL_SUCCESS) goto do_exit;

    mode = Process_->getMode(context);
    if(mode != NULL) {
        mode->setActiveQueue(queue);
        Manager_->acquireObjects(*mode);
        mode->deactivateQueue();
        mode->release();
    }

do_exit:
    cl_event *user_event = (cl_event *)user_data;
    if(user_event != NULL) delete user_event;
}
#endif

static cl_int releaseMemoryObjects(cl_command_queue command_queue)
{
    cl_int ret = CL_SUCCESS;
    cl_context context;
    ret = clGetCommandQueueInfo(command_queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, NULL);
    if(ret == CL_SUCCESS) {
        Mode *mode = Process_->getMode(context);
        if(mode != NULL) {
            mode->setActiveQueue(command_queue);
            Manager_->releaseObjects(*mode);
            mode->deactivateQueue();
            mode->decRef();
        }
    }
    return ret;
}


cl_int STD_SYMBOL(clEnqueueNDRangeKernel)(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t *global_work_offset,
    const size_t *global_work_size,
    const size_t *local_work_size,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event)
{
    ASSERTION(inGmac() == false);
    if(__opencl_clEnqueueNDRangeKernel == NULL) openclInit();
    enterGmac();
    cl_int ret = releaseMemoryObjects(command_queue);
    /* cl_event *user_event = NULL; */
    if(ret != CL_SUCCESS) goto do_exit;
    /*
    if(event == NULL) user_event = new cl_event();
    else user_event = event;
    */
    ret = __opencl_clEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset,
        global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event);
    if(ret != CL_SUCCESS) goto do_exit;
    /* ret = clSetEventCallback(*user_event, CL_COMPLETE, acquireMemoryObjects, event); */

do_exit:
    exitGmac();
    return ret;
}

cl_int STD_SYMBOL(clEnqueueTask)(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event)
{
    ASSERTION(inGmac() == false);
    if(__opencl_clEnqueueTask == NULL) openclInit();
    enterGmac();
    /* cl_event *user_event = NULL; */
    cl_int ret = releaseMemoryObjects(command_queue);
    if(ret != CL_SUCCESS) goto do_exit;
    /*
    if(event == NULL) user_event = new cl_event();
    else user_event = event;
    */
    ret = __opencl_clEnqueueTask(command_queue, kernel, num_events_in_wait_list, event_wait_list, event);
    if(ret != CL_SUCCESS) goto do_exit;
    /* ret = clSetEventCallback(*user_event, CL_COMPLETE, acquireMemoryObjects, event); */

do_exit:
    exitGmac();
    return ret;
}

cl_int STD_SYMBOL(clEnqueueNativeKernel)(
    cl_command_queue command_queue,
    void (*user_func)(void *),
    void *args,
    size_t cb_args,
    cl_uint num_mem_objects,
    const cl_mem *mem_list,
    const void **args_mem_loc,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event)
{
    ASSERTION(inGmac() == false);
    if(__opencl_clEnqueueNativeKernel == NULL) openclInit();
    /* cl_event *user_event = NULL; */
    cl_int ret = releaseMemoryObjects(command_queue);
    if(ret != CL_SUCCESS) goto do_exit;
    /*
    if(event == NULL) user_event = new cl_event();
    else user_event = event;
    */
    ret = __opencl_clEnqueueNativeKernel(command_queue, user_func, args, cb_args, num_mem_objects,
        mem_list, args_mem_loc, num_events_in_wait_list, event_wait_list, event);
    if(ret != CL_SUCCESS) goto do_exit;
    /* ret = clSetEventCallback(*user_event, CL_COMPLETE, acquireMemoryObjects, event); */

do_exit:
    exitGmac();
    return ret;
}

cl_int STD_SYMBOL(clFinish)(cl_command_queue command_queue)
{
    if(__opencl_clFinish == NULL) openclInit();
    cl_int ret = __opencl_clFinish(command_queue);
    if(inGmac() || ret != CL_SUCCESS) return ret;
    enterGmac();
    cl_context context;
    ret = clGetCommandQueueInfo(command_queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, NULL);

    if(ret == CL_SUCCESS) {
        Mode *mode = Process_->getMode(context);
        if(mode != NULL) {
            mode->setActiveQueue(command_queue);
            Manager_->acquireObjects(*mode);
            mode->deactivateQueue();
            mode->decRef();
        }
    }
    exitGmac();
    return ret;
}


cl_int APICALL clMalloc(cl_command_queue queue, void **addr, size_t count)
{
    cl_int ret = CL_SUCCESS;
    *addr = NULL;
    if(count == 0) return ret;

    enterGmac();
    gmac::trace::EnterCurrentFunction();

    size_t querySize;
    cl_context ctx;
    Mode *mode;

    ret = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, 0, NULL, &querySize);
    if(ret != CL_SUCCESS) goto exit_func;
    ret = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, querySize, &ctx, NULL);
    if(ret != CL_SUCCESS) goto exit_func;
    mode = Process_->getMode(ctx);

    if(mode != NULL) {
        count = (int(count) < getpagesize())? getpagesize(): count;
        mode->setActiveQueue(queue);
        ret = Manager_->alloc(*mode, (hostptr_t *) addr, count);
        mode->deactivateQueue();
        mode->decRef();
    }
    else ret = CL_INVALID_CONTEXT;

exit_func:
    gmac::trace::ExitCurrentFunction();
    exitGmac();
    return ret;
}

cl_int APICALL clFree(cl_command_queue queue, void *addr)
{
    cl_int ret = CL_SUCCESS;
    enterGmac();
    gmac::trace::EnterCurrentFunction();

    size_t querySize;
    cl_context ctx;
    Mode *mode;

    ret = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, 0, NULL, &querySize);
    if(ret != CL_SUCCESS) goto exit_func;
    ret = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, querySize, &ctx, NULL);
    if(ret != CL_SUCCESS) goto exit_func;
    mode = Process_->getMode(ctx);

    if(mode != NULL) {
        mode->setActiveQueue(queue);
        ret = Manager_->free(*mode, hostptr_t(addr));
        mode->deactivateQueue();
        mode->decRef();
    }
    else ret = CL_INVALID_CONTEXT;

exit_func:
    gmac::trace::ExitCurrentFunction();
    exitGmac();
    return ret;
}

cl_mem APICALL clGetBuffer(cl_context context, const void *ptr)
{
    accptr_t ret = accptr_t(0);
    enterGmac();
    Mode *mode = Process_->getMode(context);
    if(mode != NULL) {
		ret = Manager_->translate(*mode, hostptr_t(ptr));
		mode->decRef();
	}
    exitGmac();
    return ret.get();
}

#ifdef __cplusplus
}
#endif


static void openclInit()
{
    LOAD_SYM(__opencl_clCreateContext, clCreateContext);
    LOAD_SYM(__opencl_clCreateContextFromType, clCreateContextFromType);
    LOAD_SYM(__opencl_clRetainContext, clRetainContext);
    LOAD_SYM(__opencl_clReleaseContext, clReleaseContext);

    LOAD_SYM(__opencl_clCreateCommandQueue, clCreateCommandQueue);
    LOAD_SYM(__opencl_clRetainCommandQueue, clRetainCommandQueue);
    LOAD_SYM(__opencl_clReleaseCommandQueue, clReleaseCommandQueue);

    LOAD_SYM(__opencl_clEnqueueNDRangeKernel, clEnqueueNDRangeKernel);
    LOAD_SYM(__opencl_clEnqueueTask, clEnqueueTask);
    LOAD_SYM(__opencl_clEnqueueNativeKernel, clEnqueueNativeKernel);

    LOAD_SYM(__opencl_clFinish, clFinish);
}

void initGmac()
{
    /* TRACE cannot be called until TLS::Init() has been executed;
	 * TLS::Init() is part of the Process construction
	 */
    Process_ = new __impl::opencl::lite::Process();

	TRACE(GLOBAL, "Initializing Memory Manager");
    __impl::memory::Handler::setEntry(enterGmac);
    __impl::memory::Handler::setExit(exitGmac);
    Manager_ = new gmac::memory::Manager(*Process_);
}


namespace __impl {
    namespace core {
        Mode &getMode(Mode &mode) { return mode; }
        Process &getProcess() { return *Process_; }
    }
    namespace memory {
        Manager &getManager() { return *Manager_; }
    }
}


#if defined(_WIN32)
#include <windows.h>


// DLL entry function (called on load, unload, ...)
BOOL APIENTRY DllMain(HANDLE /*hModule*/, DWORD dwReason, LPVOID /*lpReserved*/)
{
    switch(dwReason) {
        case DLL_PROCESS_ATTACH:
            openclInit();
            break;
        case DLL_PROCESS_DETACH:
            break;
        case DLL_THREAD_ATTACH:
			isRunTimeThread_.set(&privateFalse);
            break;
        case DLL_THREAD_DETACH:
            break;
    };
    return TRUE;
}

#else

SYM(int, pthread_create__, pthread_t *__restrict, __const pthread_attr_t *, void *(*)(void *), void *);

void threadInit(void)
{
    LOAD_SYM(pthread_create__, pthread_create);
}

static void __attribute__((destructor())) gmacPthreadFini(void)
{
}

struct gmac_thread_t {
    void *(*start_routine)(void *);
    void *arg;
    bool externCall;
};

static void *gmac_pthread(void *arg)
{
    gmac::trace::StartThread("CPU");

    gmac_thread_t *gthread = (gmac_thread_t *)arg;
    bool externCall = gthread->externCall;

    // This TLS variable is necessary before entering GMAC
    if (externCall != true) {
        isRunTimeThread_.set(&privateTrue);
    } else {
        isRunTimeThread_.set(&privateFalse);
    }

    enterGmac();

    gmac::trace::SetThreadState(gmac::trace::Running);
    if(externCall) exitGmac();
    void *ret = gthread->start_routine(gthread->arg);
    if(externCall) enterGmac();

    // Modes and Contexts already destroyed in Process destructor
    free(gthread);
    gmac::trace::SetThreadState(gmac::trace::Idle);
    exitGmac();
    return ret;
}

int pthread_create(pthread_t *__restrict newthread,
                   __const pthread_attr_t *__restrict attr,
                   void *(*start_routine)(void *),
                   void *__restrict arg)
{
    int ret = 0;
    bool externCall = inGmac() == 0;
    if(externCall) enterGmac();
    TRACE(GLOBAL, "New POSIX thread");
    gmac_thread_t *gthread = (gmac_thread_t *)malloc(sizeof(gmac_thread_t));
    gthread->start_routine = start_routine;
    gthread->arg = arg;
    gthread->externCall = externCall;
    ret = pthread_create__(newthread, attr, gmac_pthread, (void *)gthread);
    if(externCall) exitGmac();
    return ret;
}


#endif
