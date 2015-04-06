#include "IOBuffer.h"
#include "Mode.h"

#include "hpe/Mode.h"

namespace __impl { namespace cuda {

gmacError_t
IOBuffer::wait(bool internal)
{
    EventMap::iterator it;
    it = map_.find(mode_);
    ASSERTION(state_ == Idle || it != map_.end());

    gmacError_t ret = gmacSuccess;

    if (state_ != Idle) {
        ASSERTION(mode_ != NULL);
        ASSERTION(started_ == true);
        CUevent start = it->second.start;
        CUevent end   = it->second.end;
        trace::SetThreadState(trace::Wait);
        ret = mode_->waitForEvent(end, internal);
        trace::SetThreadState(trace::Running);
        if(state_ == ToHost) DataCommToHost(*mode_, start, end, xfer_, it->second.time);
        else if(state_ == ToAccelerator) DataCommToAccelerator(*mode_, start, end, xfer_, it->second.time);
        TRACE(LOCAL,"Buffer %p goes Idle", this);
        state_ = Idle;
        mode_  = NULL;
        started_ = false;
    } else {
        ASSERTION(mode_ == NULL);
    }

    return ret;
}


}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
