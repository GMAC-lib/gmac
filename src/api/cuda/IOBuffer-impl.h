#ifndef GMAC_API_CUDA_IOBUFFER_IMPL_H_
#define GMAC_API_CUDA_IOBUFFER_IMPL_H_

#include "Tracer.h"

namespace __impl { namespace cuda {

inline void
IOBuffer::toHost(Mode &mode, CUstream s)
{
    CUresult ret = CUDA_SUCCESS;

    EventMap::iterator it;
    it = map_.find(&mode);
    if (it == map_.end()) {
        CUevent start;
        CUevent end;
        ret = cuEventCreate(&start, CU_EVENT_DEFAULT);
        ASSERTION(ret == CUDA_SUCCESS);
        ret = cuEventCreate(&end, CU_EVENT_DEFAULT);
        ASSERTION(ret == CUDA_SUCCESS);

        stamp_t stamp;
        stamp.start = start;
        stamp.end = end;
        stamp.time = 0;

        std::pair<EventMap::iterator, bool> par =
            map_.insert(EventMap::value_type(&mode, stamp));
        it = par.first;
    }

    trace::TimeMark(it->second.time);
    ret = cuEventRecord(it->second.start, s);
    ASSERTION(ret == CUDA_SUCCESS);
    state_  = ToHost;
    TRACE(LOCAL,"Buffer %p goes toHost", this); 
    stream_ = s;
    mode_   = &mode;
}

inline void
IOBuffer::toAccelerator(Mode &mode, CUstream s)
{
    CUresult ret = CUDA_SUCCESS;

    EventMap::iterator it;
    it = map_.find(&mode);
    if (it == map_.end()) {
        CUevent start;
        CUevent end;
        ret = cuEventCreate(&start, CU_EVENT_DEFAULT);
        ASSERTION(ret == CUDA_SUCCESS);
        ret = cuEventCreate(&end, CU_EVENT_DEFAULT);
        ASSERTION(ret == CUDA_SUCCESS);

        stamp_t stamp;
        stamp.start = start;
        stamp.end = end;
        stamp.time = 0;

        std::pair<EventMap::iterator, bool> ret =
            map_.insert(EventMap::value_type(&mode, stamp));
        it = ret.first;
    }

    trace::TimeMark(it->second.time);
    ret = cuEventRecord(it->second.start, s);
    ASSERTION(ret == CUDA_SUCCESS);
    state_  = ToAccelerator;
    TRACE(LOCAL,"Buffer %p goes toAccelerator", this);
    stream_ = s;
    mode_   = &mode;
}

inline void
IOBuffer::started(size_t size)
{
    EventMap::iterator it;
    it = map_.find(mode_);
    ASSERTION(started_ == false);
    ASSERTION(state_ != Idle && it != map_.end());
    CUresult ret = cuEventRecord(it->second.end, stream_);
    ASSERTION(ret == CUDA_SUCCESS);
    started_ = true;
    xfer_ = size;
}

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
