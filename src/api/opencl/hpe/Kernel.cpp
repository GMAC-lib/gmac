#include "Kernel.h"
#include "Mode.h"
#include "Accelerator.h"

#include "trace/Tracer.h"

namespace __impl { namespace opencl { namespace hpe {

KernelLaunch::MapGlobalSubBuffer KernelLaunch::mapSubBuffer_;

gmacError_t
KernelLaunch::execute()
{
    trace_.init((THREAD_T)mode_.getId());
    size_t *globalWorkSize   = globalWorkSize_;
    size_t *localWorkSize    = workLocalDim_ > 0? localWorkSize_: NULL;
    size_t *globalWorkOffset = offsetDim_    > 0? globalWorkOffset_: NULL;

    gmacError_t ret = dynamic_cast<Mode &>(mode_).getAccelerator().execute(stream_, f_, workGlobalDim_,
        globalWorkOffset, globalWorkSize, localWorkSize, &event_);
    if(ret == gmacSuccess) trace_.trace(f_, event_);
    return ret;
}

cl_mem
KernelLaunch::getSubBuffer(__impl::opencl::hpe::Mode &mode, hostptr_t ptr, accptr_t accPtr, size_t size)
{
    cl_mem ret;

    CacheSubBuffer::const_iterator itCacheMap = cacheSubBuffer_.find(ptr);
    if (itCacheMap == cacheSubBuffer_.end()) {
        // find always returns a value
        MapGlobalSubBuffer::iterator itGlobalMap = mapSubBuffer_.findMode(mode);

        MapSubBuffer &mapMode = itGlobalMap->second;
        MapSubBuffer::iterator itModeMap = mapMode.find(ptr);

        if (itModeMap == mapMode.end()) {
            int err;
            cl_buffer_region region;
            region.origin = accPtr.offset();
            region.size   = size - accPtr.offset();
            ret = clCreateSubBuffer(accPtr.get(), CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
            ASSERTION(err == CL_SUCCESS);

            mapMode.insert(MapSubBuffer::value_type(ptr, CLMemRef(ret, 0)));
            itModeMap = mapMode.find(ptr);
        }

        ret = itModeMap->second.first;
        itModeMap->second.second++;
        cacheSubBuffer_.insert(CacheSubBuffer::value_type(ptr, CacheEntry(&mode, itModeMap)));
    } else {
        // Cache-entry -> CacheEntry -> iterator -> pair
        ret = itCacheMap->second.second->second.first;
    }

    return ret;
}



}}}
