#ifndef GMAC_API_CUDA_HPE_KERNEL_IMPL_H_
#define GMAC_API_CUDA_HPE_KERNEL_IMPL_H_

#include "util/Logger.h"

namespace __impl { namespace cuda { namespace hpe {

inline
Argument::Argument(const void * ptr, size_t size, long_t offset) :
    ptr_(ptr), size_(size), offset_(offset)
{
}

inline
KernelLaunch *
Kernel::launch(Mode &mode, KernelConfig & c)
{
    KernelLaunch * l = new cuda::hpe::KernelLaunch(mode, *this, c);
    return l;
}

inline
KernelConfig::KernelConfig() :
    argsSize_(0)
{
    stack_ = new uint8_t[StackSize_];
}

inline
KernelConfig::~KernelConfig()
{
    delete [] stack_;
}

inline
void KernelConfig::pushArgument(const void *arg, size_t size, long_t offset)
{
    ASSERTION(offset + size < KernelConfig::StackSize_);

    memcpy(&stack_[offset], arg, size);
    push_back(Argument(&stack_[offset], size, offset));
    argsSize_ = size_t(offset) + size;
}

inline size_t
KernelConfig::argsSize() const
{
    return argsSize_;
}

inline uint8_t *
KernelConfig::argsArray()
{
    return stack_;
}

inline CUevent
KernelLaunch::getCUevent()
{
    return end_;
}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
