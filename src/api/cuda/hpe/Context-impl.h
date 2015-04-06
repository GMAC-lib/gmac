#ifndef GMAC_API_CUDA_HPE_CONTEXT_IMPL_H_
#define GMAC_API_CUDA_HPE_CONTEXT_IMPL_H_

#include "Accelerator.h"
#include "Kernel.h"

namespace __impl { namespace cuda { namespace hpe {

inline gmacError_t
Context::call(dim3 Dg, dim3 Db, size_t shared, cudaStream_t tokens)
{
    call_ = KernelConfig(Dg, Db, shared, tokens, streamLaunch_);
    // TODO: do some checking
    return gmacSuccess;
}

inline gmacError_t
Context::argument(const void *arg, size_t size, off_t offset)
{
    call_.pushArgument(arg, size, offset);
    // TODO: do some checking
    return gmacSuccess;
}

inline const stream_t
Context::eventStream() const
{
    return streamLaunch_;
}

inline Accelerator &
Context::accelerator()
{
    return dynamic_cast<Accelerator &>(acc_);
}

}}}

#endif
