#ifdef USE_DBC

#include "core/hpe/Mode.h"

namespace __dbc { namespace core { namespace hpe {

void
Mode::cleanUpContexts()
{
    Parent::cleanUpContexts();
}

gmacError_t
Mode::cleanUp()
{
    gmacError_t ret;
    ret = Parent::cleanUp();
    return ret;
}

Mode::~Mode()
{
}

Mode::Mode(ProcessImpl &proc, AcceleratorImpl &acc, AddressSpaceImpl &aSpace) :
    Parent(proc, acc, aSpace)
{
}


gmacError_t
Mode::map(accptr_t &dst, hostptr_t src, size_t size, unsigned align)
{
    REQUIRES(size > 0);

    gmacError_t ret;
    ret = Parent::map(dst, src, size, align);

    return ret;
}

gmacError_t
Mode::unmap(hostptr_t addr, size_t size)
{
    REQUIRES(addr != NULL);

    gmacError_t ret;
    ret = Parent::unmap(addr, size);

    return ret;
}

gmacError_t
Mode::copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size)
{
    REQUIRES(acc != 0);
    REQUIRES(host != NULL);
    REQUIRES(size > 0);

    gmacError_t ret;
    ret = Parent::copyToAccelerator(acc, host, size);

    return ret;
}

gmacError_t
Mode::copyToHost(hostptr_t host, const accptr_t acc, size_t size)
{
    REQUIRES(host != NULL);
    REQUIRES(acc != 0);
    REQUIRES(size > 0);

    gmacError_t ret;
    ret = Parent::copyToHost(host, acc, size);

    return ret;
}

gmacError_t
Mode::copyAccelerator(accptr_t dst, const accptr_t src, size_t size)
{
    REQUIRES(dst != 0);
    REQUIRES(src != 0);
    REQUIRES(size > 0);

    gmacError_t ret;
    ret = Parent::copyAccelerator(dst, src, size);

    return ret;
}

gmacError_t
Mode::memset(accptr_t addr, int c, size_t size)
{
    REQUIRES(addr != 0);
    REQUIRES(size > 0);

    gmacError_t ret;
    ret = Parent::memset(addr, c, size);

    return ret;
}

gmacError_t
Mode::bufferToAccelerator(accptr_t dst, IOBufferImpl &buffer, size_t size, size_t off)
{
    REQUIRES(size > 0);
    REQUIRES(off + size <= buffer.size());

    gmacError_t ret;

    ret = Parent::bufferToAccelerator(dst, buffer, size, off);

    return ret;
}

gmacError_t
Mode::acceleratorToBuffer(IOBufferImpl &buffer, const accptr_t dst, size_t size, size_t off)
{
    REQUIRES(size > 0);
    REQUIRES(off + size <= buffer.size());

    gmacError_t ret;

    ret = Parent::acceleratorToBuffer(buffer, dst, size, off);

    return ret;
}

void
Mode::registerKernel(gmac_kernel_id_t k, KernelImpl &kernel)
{
    REQUIRES(kernels_.find(k) == kernels_.end());

    Parent::registerKernel(k, kernel);

    ENSURES(kernels_.find(k) != kernels_.end());
}

std::string
Mode::getKernelName(gmac_kernel_id_t k) const
{
    REQUIRES(kernels_.find(k) != kernels_.end());

    std::string ret = Parent::getKernelName(k);

    return ret;
}

gmacError_t 
Mode::moveTo(__impl::core::hpe::Accelerator &acc)
{
    REQUIRES(&acc != acc_);

    gmacError_t ret;
    ret = Parent::moveTo(acc);

    return ret;
}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
