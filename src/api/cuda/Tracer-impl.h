#ifndef GMAC_API_CUDA_TRACER_IMPL_H_
#define GMAC_API_CUDA_TRACER_IMPL_H_

#include "Mode.h"

namespace __impl { namespace cuda { 

inline void DataCommunication(Mode &mode, THREAD_T src, THREAD_T dst, CUevent start,
        CUevent end, size_t size, uint64_t t)
{
#if defined(USE_TRACE)
    uint64_t delta = 0;
    gmacError_t ret = mode.eventTime(delta, start, end);
    ASSERTION(ret == gmacSuccess);
    return trace::DataCommunication(src, dst, delta, size, t);
#endif
}

inline void DataCommToAccelerator(Mode &mode, CUevent start, CUevent end, size_t size, uint64_t t)
{
#if defined(USE_TRACE)
    return DataCommunication(mode, trace::GetThreadId(), dynamic_cast<core::Mode &>(mode).getId(), start, end, size, t);
#endif
}

inline void DataCommToHost(Mode &mode, CUevent start, CUevent end, size_t size, uint64_t t)
{
#if defined(USE_TRACE)
    return DataCommunication(mode, dynamic_cast<core::Mode &>(mode).getId(), trace::GetThreadId(), start, end, size, t);
#endif
}


}}

#endif
