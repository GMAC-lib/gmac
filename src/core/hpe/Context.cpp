#include "memory/Manager.h"
#include "trace/Tracer.h"

#include "core/hpe/Accelerator.h"
#include "core/hpe/Mode.h"
#include "core/hpe/Context.h"

namespace __impl { namespace core { namespace hpe {

Context::Context(Mode &mode, stream_t streamLaunch, stream_t streamToAccelerator, stream_t streamToHost, stream_t streamAccelerator) :
    gmac::util::RWLock("Context"),
    acc_(mode.getAccelerator()),
    mode_(mode),
    streamLaunch_(streamLaunch),
    streamToAccelerator_(streamToAccelerator),
    streamToHost_(streamToHost),
    streamAccelerator_(streamAccelerator),
    bufferWrite_(NULL),
    bufferRead_(NULL)
{
}

Context::~Context()
{
}

void
Context::init()
{
}

gmacError_t Context::copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size)
{
    TRACE(LOCAL,"Transferring "FMT_SIZE" bytes from host %p to accelerator %p", size, host, acc.get());
    trace::EnterCurrentFunction();
    if(size == 0) return gmacSuccess; /* Fast path */
    /* In case there is no page-locked memory available, use the slow path */
    if(bufferWrite_ == NULL) bufferWrite_ = &static_cast<IOBuffer &>(mode_.createIOBuffer(util::params::ParamBlockSize, GMAC_PROT_WRITE));
    if(bufferWrite_->async() == false) {
        mode_.destroyIOBuffer(*bufferWrite_);
        bufferWrite_ = NULL;
        TRACE(LOCAL,"Not using pinned memory for transfer");
        gmacError_t ret = acc_.copyToAccelerator(acc, host, size, mode_);
        trace::ExitCurrentFunction();
        return ret;
    }
    gmacError_t ret = bufferWrite_->wait(true);
    ptroff_t offset = 0;
    while(size_t(offset) < size) {
        ret = bufferWrite_->wait(true);
        if(ret != gmacSuccess) break;
        ptroff_t len = ptroff_t(bufferWrite_->size());
        if((size - offset) < bufferWrite_->size()) len = ptroff_t(size - offset);
        trace::EnterCurrentFunction();
        ::memcpy(bufferWrite_->addr(), host + offset, len);
        trace::ExitCurrentFunction();
        ASSERTION(size_t(len) <= util::params::ParamBlockSize);
        ret = acc_.copyToAcceleratorAsync(acc + offset, *bufferWrite_, 0, len, mode_, streamToAccelerator_);
        ASSERTION(ret == gmacSuccess);
        if(ret != gmacSuccess) break;
        offset += len;
    }
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Context::copyToHost(hostptr_t host, const accptr_t acc, size_t size)
{
    TRACE(LOCAL,"Transferring "FMT_SIZE" bytes from accelerator %p to host %p", size, acc.get(), host);
    trace::EnterCurrentFunction();
    if(size == 0) return gmacSuccess;
    if(bufferRead_ == NULL) bufferRead_ = &static_cast<IOBuffer &>(mode_.createIOBuffer(util::params::ParamBlockSize, GMAC_PROT_READ));
    if(bufferRead_->async() == false) {
        mode_.destroyIOBuffer(*bufferRead_);
        bufferRead_ = NULL;
        TRACE(LOCAL,"Not using pinned memory for transfer");
        gmacError_t ret = acc_.copyToHost(host, acc, size, mode_);
        trace::ExitCurrentFunction();
        return ret;
    }

    gmacError_t ret = bufferRead_->wait(true);
    if(ret != gmacSuccess) { trace::ExitCurrentFunction(); return ret; }
    ptroff_t offset = 0;
    while(size_t(offset) < size) {
        ptroff_t len = ptroff_t(bufferRead_->size());
        if((size - offset) < bufferRead_->size()) len = ptroff_t(size - offset);
        ret = acc_.copyToHostAsync(*bufferRead_, 0, acc + offset, len, mode_, streamToHost_);
        ASSERTION(ret == gmacSuccess);
        if(ret != gmacSuccess) break;
        ret = bufferRead_->wait(true);
        if(ret != gmacSuccess) break;
        trace::EnterCurrentFunction();
        ::memcpy((uint8_t *)host + offset, bufferRead_->addr(), len);
        trace::ExitCurrentFunction();
        offset += len;
    }
    trace::ExitCurrentFunction();
    return ret;
}

}}}
