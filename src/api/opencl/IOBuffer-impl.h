#ifndef GMAC_API_OPENCL_IOBUFFER_IMPL_H_
#define GMAC_API_OPENCL_IOBUFFER_IMPL_H_

#include "Tracer.h"

namespace __impl { namespace opencl {

inline
IOBuffer::IOBuffer(Mode &mode, hostptr_t addr, size_t size, cl_mem mem, GmacProtection prot) :
    gmac::core::IOBuffer(addr, size, mem != NULL, prot),
    mem_(mem),
    event_(NULL),
    mode_(NULL),
    started_(false)
{
}

inline void
IOBuffer::toHost(Mode &mode)
{
    ASSERTION(started_ == false);
    ASSERTION(state_ == Idle);
    state_  = ToHost;
    TRACE(LOCAL,"Buffer %p goes toHost", this);
    mode_   = &mode;
    trace_.init(trace_.getModeId(*mode_), trace_.getThreadId());
}

inline void
IOBuffer::toAccelerator(Mode &mode)
{
    ASSERTION(started_ == false);
    ASSERTION(state_ == Idle);
    state_  = ToAccelerator;
    TRACE(LOCAL,"Buffer %p goes toAccelerator", this);
    mode_   = &mode;
    trace_.init(trace_.getThreadId(), trace_.getModeId(*mode_));
}

inline void
IOBuffer::started(cl_event event, size_t size)
{
    started(event, event, size);
}

inline void
IOBuffer::started(cl_event start, cl_event end, size_t size)
{
    TRACE(LOCAL,"Buffer %p starts", this);
    ASSERTION(started_ == false);
    ASSERTION(mode_ != NULL);
    start_ = start;
    event_ = end;
    started_ = true;
    last_ = size;
}

inline gmacError_t
IOBuffer::wait(bool internal)
{
    TRACE(LOCAL,"Buffer %p waits: %d", this, state_ == Idle || started_ == true);
    ASSERTION(state_ == Idle || started_ == true);

    gmacError_t ret = gmacSuccess;

    if (state_ != Idle) {
        ASSERTION(mode_ != NULL);
        ASSERTION(event_ != NULL);

        trace::SetThreadState(trace::Wait);
        ret = mode_->waitForEvent(event_);
        trace::SetThreadState(trace::Running);
        ASSERTION(ret == gmacSuccess);
        trace_.trace(start_, event_, last_);
                cl_int clret = CL_SUCCESS;
                if(event_ != start_) clret = clReleaseEvent(start_);
        clret |= clReleaseEvent(event_);
        ASSERTION(clret == CL_SUCCESS);
        TRACE(LOCAL,"Buffer %p goes Idle", this);

        state_ = Idle;
        event_ = NULL;
        mode_  = NULL;
        started_ = false;
    } else {
        ASSERTION(mode_ == NULL);
    }

    return ret;
}

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
