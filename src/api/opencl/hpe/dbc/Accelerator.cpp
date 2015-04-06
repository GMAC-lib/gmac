#ifdef USE_DBC
#include "api/opencl/hpe/Accelerator.h"
#include "api/opencl/IOBuffer.h"

#include "core/hpe/Mode.h"

namespace __dbc { namespace opencl { namespace hpe {

Accelerator::Accelerator(int n, cl_context context, cl_device_id device, unsigned major, unsigned minor) :
    __impl::opencl::hpe::Accelerator(n, context, device, major, minor)
{
    REQUIRES(n >= 0);
}

Accelerator::~Accelerator()
{
}

gmacError_t Accelerator::unmap(hostptr_t host, size_t size)
{
    REQUIRES(host != NULL);
    REQUIRES(size > 0);
    accptr_t addr;
    size_t s;
    bool hasMapping = allocations_.find(host, addr, s);
    ENSURES(hasMapping == true);
    ENSURES(s == size);
    return __impl::opencl::hpe::Accelerator::unmap(host, size);
}

gmacError_t Accelerator::copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size, __impl::core::hpe::Mode &mode)
{
    // PRECONDITIONS
    REQUIRES(acc  != 0);
    REQUIRES(host != NULL);
    REQUIRES(size > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::opencl::hpe::Accelerator::copyToAccelerator(acc, host, size, mode);
    // POSTCONDITIONS
    ENSURES(ret == gmacSuccess);

    return ret;
}

gmacError_t Accelerator::copyToHost(hostptr_t host, const accptr_t acc, size_t size, __impl::core::hpe::Mode &mode)
{
    // PRECONDITIONS
    REQUIRES(host != NULL);
    REQUIRES(acc  != 0);
    REQUIRES(size > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::opencl::hpe::Accelerator::copyToHost(host, acc, size, mode);
    // POSTCONDITIONS
    ENSURES(ret == gmacSuccess);

    return ret;
}

gmacError_t Accelerator::copyAccelerator(accptr_t dst, const accptr_t src, size_t size, stream_t stream)
{
    // PRECONDITIONS
    REQUIRES(src != 0);
    REQUIRES(dst != 0);
    REQUIRES(size > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::opencl::hpe::Accelerator::copyAccelerator(dst, src, size, stream);
    // POSTCONDITIONS
    ENSURES(ret == gmacSuccess);

    return ret;
}


}}}
#endif
/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
