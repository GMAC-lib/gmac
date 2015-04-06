/**
 * \file src/hpe/gmac.cpp
 *
 * Implementation of the generic HPE API calls
 */

#include <cstdlib>

#ifdef USE_CUDA
#include "include/gmac/cuda.h"
#else
#include "include/gmac/opencl.h"
#endif

#include "config/config.h"
#include "config/order.h"

#include "util/Atomics.h"
#include "util/Logger.h"

#include "core/IOBuffer.h"

#include "core/hpe/Accelerator.h"
#include "core/hpe/Mode.h"
#include "core/hpe/Kernel.h"
#include "core/hpe/Process.h"
#include "core/hpe/Thread.h"

#include "memory/Manager.h"
#include "memory/Allocator.h"
#ifdef DEBUG
#include "memory/protocol/common/BlockState.h"
#endif

#include "trace/Tracer.h"

#if defined(GMAC_DLL)
#include "init.h"
#endif

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

using namespace __impl::core::hpe;
using namespace __impl::memory;

using __impl::util::params::ParamBlockSize;
using __impl::util::params::ParamAutoSync;

GMAC_API unsigned APICALL
gmacGetNumberOfAccelerators()
{
    unsigned ret;
    enterGmac();
    gmac::trace::EnterCurrentFunction();
    ret = unsigned(getProcess().nAccelerators());
    gmac::trace::ExitCurrentFunction();
    exitGmac();
    return ret;
}

GMAC_API unsigned APICALL
gmacGetCurrentAcceleratorId()
{
    unsigned ret;
    enterGmac();
    gmac::trace::EnterCurrentFunction();
    ret = Thread::getCurrentMode().getAccelerator().id();;
    gmac::trace::ExitCurrentFunction();
    exitGmac();
    return ret;
}

GMAC_API gmacError_t APICALL
gmacGetAcceleratorInfo(unsigned acc, GmacAcceleratorInfo *info)
{
    enterGmacExclusive();
    gmac::trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;
    __impl::core::hpe::Process &process = getProcess();
    if (acc < process.nAccelerators() && info != NULL) {
        Accelerator &accelerator = process.getAccelerator(acc);
        accelerator.getAcceleratorInfo(*info);
    } else {
        ret = gmacErrorInvalidValue;
    }
    gmac::trace::ExitCurrentFunction();
    Thread::setLastError(ret);
    exitGmac();
    return ret;

}

GMAC_API gmacError_t APICALL
gmacGetFreeMemory(unsigned acc, size_t *freeMemory)
{
    enterGmacExclusive();
    gmac::trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;
    __impl::core::hpe::Process &process = getProcess();
    if (acc < process.nAccelerators() && freeMemory != NULL) {
        size_t total;
        Accelerator &accelerator = process.getAccelerator(acc);
        accelerator.getMemInfo(*freeMemory, total);
    } else {
        ret = gmacErrorInvalidValue;
    }
    gmac::trace::ExitCurrentFunction();
    Thread::setLastError(ret);
    exitGmac();
    return ret;
}

GMAC_API gmacError_t APICALL
gmacMigrate(unsigned acc)
{
    gmacError_t ret = gmacSuccess;
    enterGmacExclusive();
    gmac::trace::EnterCurrentFunction();
    ret = getProcess().migrate(acc);
    gmac::trace::ExitCurrentFunction();
    Thread::setLastError(ret);
    exitGmac();
    return ret;
}


GMAC_API gmacError_t APICALL
gmacMemoryMap(void *cpuPtr, size_t count, GmacProtection prot)
{
#if 0
    gmacError_t ret = gmacSuccess;
    if (count == 0) {
        return ret;
    }
        enterGmac();
    gmac::trace::EnterCurrentFunction();
    // TODO Remove alignment constraints?
    count = (int(count) < getpagesize())? getpagesize(): count;
    ret = getManager().map(cpuPtr, count, prot);
    gmac::trace::ExitCurrentFunction();
        exitGmac();
    return ret;
#endif
    gmacError_t ret = gmacErrorFeatureNotSupported;
    Thread::setLastError(ret);
    return ret;
}


GMAC_API gmacError_t APICALL
gmacMemoryUnmap(void *cpuPtr, size_t count)
{
#if 0
    gmacError_t ret = gmacSuccess;
    if (count == 0) {
        return ret;
    }
        enterGmac();
    gmac::trace::EnterCurrentFunction();
    // TODO Remove alignment constraints?
    count = (int(count) < getpagesize())? getpagesize(): count;
    ret = getManager().unmap(cpuPtr, count);
    gmac::trace::ExitCurrentFunction();
        exitGmac();
    return ret;
#endif
    gmacError_t ret = gmacErrorFeatureNotSupported;
    Thread::setLastError(ret);
    return ret;
}


GMAC_API gmacError_t APICALL
gmacMalloc(void **cpuPtr, size_t count)
{
    gmacError_t ret = gmacSuccess;
    if (count == 0) {
        *cpuPtr = NULL;
        Thread::setLastError(ret);
        return ret;
    }
    enterGmac();
    gmac::trace::EnterCurrentFunction();
    if(hasAllocator() && count < (ParamBlockSize / 2)) {
        *cpuPtr = getAllocator().alloc(Thread::getCurrentMode(), count, hostptr_t(RETURN_ADDRESS));
    }
    else {
        count = (int(count) < getpagesize())? getpagesize(): count;
        ret = getManager().alloc(Thread::getCurrentMode(), (hostptr_t *) cpuPtr, count);
    }
    gmac::trace::ExitCurrentFunction();
    Thread::setLastError(ret);
    exitGmac();
    return ret;
}

GMAC_API gmacError_t APICALL
gmacGlobalMalloc(void **cpuPtr, size_t count, GmacGlobalMallocType hint)
{
    gmacError_t ret = gmacSuccess;
    if(count == 0) {
        *cpuPtr = NULL;
        Thread::setLastError(ret);
        return ret;
    }
    enterGmac();
    gmac::trace::EnterCurrentFunction();
    count = (count < (size_t)getpagesize()) ? (size_t)getpagesize(): count;
    ret = getManager().globalAlloc(Thread::getCurrentMode(), (hostptr_t *)cpuPtr, count, hint);
    gmac::trace::ExitCurrentFunction();
    Thread::setLastError(ret);
    exitGmac();
    return ret;
}

GMAC_API gmacError_t APICALL
gmacFree(void *cpuPtr)
{
    gmacError_t ret = gmacSuccess;
    enterGmac();
    if(cpuPtr == NULL) {
        Thread::setLastError(ret);
        exitGmac();
        return ret;
    }
    gmac::trace::EnterCurrentFunction();
    __impl::core::hpe::Mode &mode = Thread::getCurrentMode();
    if(hasAllocator() == false || getAllocator().free(mode, hostptr_t(cpuPtr)) == false) {
        ret = getManager().free(mode, hostptr_t(cpuPtr));
    }
    gmac::trace::ExitCurrentFunction();
    Thread::setLastError(ret);
    exitGmac();
    return ret;
}

GMAC_API __gmac_accptr_t APICALL
gmacPtr(const void *ptr)
{
    accptr_t ret = accptr_t(0);
    enterGmac();
    ret = getManager().translate(Thread::getCurrentMode(), hostptr_t(ptr));
    exitGmac();
    TRACE(GLOBAL, "Translate %p to %p", ptr, ret.get());
    return ret.get();
}

gmacError_t GMAC_LOCAL
gmacLaunch(__impl::core::hpe::KernelLaunch &launch)
{
    gmacError_t ret = gmacSuccess;
    __impl::core::hpe::Mode &mode = launch.getMode();
    Manager &manager = getManager();
    TRACE(GLOBAL, "Flush the memory used in the kernel");
    const std::list<__impl::memory::ObjectInfo> &objects = launch.getObjects();
    // If the launch object does not contain objects, assume all the objects
    // in the mode are released
    ret = manager.releaseObjects(mode, objects);
    CFATAL(ret == gmacSuccess, "Error releasing objects");

    TRACE(GLOBAL, "Kernel Launch");
    ret = mode.execute(launch);

    if (ParamAutoSync == true) {
        TRACE(GLOBAL, "Waiting for Kernel to complete");
        mode.wait();
        TRACE(GLOBAL, "Memory Sync");
        ret = manager.acquireObjects(Thread::getCurrentMode(), objects);
        CFATAL(ret == gmacSuccess, "Error waiting for kernel");
    }

    Thread::setLastError(ret);

    return ret;
}

GMAC_API gmacError_t APICALL
gmacLaunch(gmac_kernel_id_t k)
{
    enterGmac();
    gmac::trace::EnterCurrentFunction();
    __impl::core::hpe::Mode &mode = Thread::getCurrentMode();
    __impl::core::hpe::KernelLaunch *launch = NULL;
    gmacError_t ret = mode.launch(k, launch);

    if(ret == gmacSuccess) {
        ret = gmacLaunch(*launch);
        delete launch;
    }

    gmac::trace::ExitCurrentFunction();

    Thread::setLastError(ret);
    exitGmac();

    return ret;
}

gmacError_t GMAC_LOCAL
gmacThreadSynchronize(__impl::core::hpe::KernelLaunch &launch)
{
    gmacError_t ret = gmacSuccess;
    if(ParamAutoSync == false) {
        __impl::core::hpe::Mode &mode = Thread::getCurrentMode();
        mode.wait(launch);
        TRACE(GLOBAL, "Memory Sync");
        ret = getManager().acquireObjects(mode, launch.getObjects());
    }
    return ret;
}

GMAC_API gmacError_t APICALL
gmacThreadSynchronize()
{
    enterGmac();
    gmac::trace::EnterCurrentFunction();

    gmacError_t ret = gmacSuccess;
    if (ParamAutoSync == false) {
        __impl::core::hpe::Mode &mode = Thread::getCurrentMode();
        mode.wait();
        TRACE(GLOBAL, "Memory Sync");
        ret = getManager().acquireObjects(mode);
    }

    gmac::trace::ExitCurrentFunction();
    Thread::setLastError(ret);
    exitGmac();
    return ret;
}

GMAC_API gmacError_t APICALL
gmacGetLastError()
{
    enterGmac();
    gmacError_t ret = Thread::getLastError();
    exitGmac();
    return ret;
}

/** \todo Move to a more CUDA-like API */
GMAC_API void * APICALL
gmacMemset(void *s, int c, size_t size)
{
    enterGmac();
    void *ret = s;
    getManager().memset(Thread::getCurrentMode(), hostptr_t(s), c, size);
    exitGmac();
    return ret;
}

/** \todo Move to a more CUDA-like API */
GMAC_API void * APICALL
gmacMemcpy(void *dst, const void *src, size_t size)
{
    enterGmac();
    void *ret = dst;

    // Locate memory regions (if any)
    Process &proc = getProcess();
    __impl::core::Mode *dstMode = proc.owner(hostptr_t(dst), size);
    __impl::core::Mode *srcMode = proc.owner(hostptr_t(src), size);
    if (dstMode == NULL && srcMode == NULL) {
        exitGmac();
        return ::memcpy(dst, src, size);
    }
    getManager().memcpy(Thread::getCurrentMode(), hostptr_t(dst), hostptr_t(src), size);

    exitGmac();
    return ret;
}

GMAC_API gmacError_t APICALL
gmacSetAddressSpace(unsigned aSpaceId)
{
    enterGmac();
    gmacError_t ret = getProcess().setAddressSpace(Thread::getCurrentMode(), aSpaceId);
    exitGmac();
    return ret;
}

/** \todo Return error */
GMAC_API void APICALL
gmacSend(THREAD_T id)
{
    enterGmac();
    getProcess().send((THREAD_T)id);
    exitGmac();
}

/** \todo Return error */
GMAC_API void APICALL
gmacReceive()
{
    enterGmac();
    getProcess().receive();
    exitGmac();
}

/** \todo Return error */
GMAC_API void APICALL
gmacSendReceive(THREAD_T id)
{
    enterGmac();
    getProcess().sendReceive((THREAD_T)id);
    exitGmac();
}

/** \todo Return error */
GMAC_API void APICALL
gmacCopy(THREAD_T id)
{
    enterGmac();
    getProcess().copy((THREAD_T)id);
    exitGmac();
}

#ifdef USE_INTERNAL_API

GMAC_API gmacError_t APICALL
__gmacFlushDirty()
{
    enterGmac();
    gmacError_t ret = getManager().flushDirty(Thread::getCurrentMode());
    Thread::setLastError(ret);
    exitGmac();
    return ret;
}

#endif
