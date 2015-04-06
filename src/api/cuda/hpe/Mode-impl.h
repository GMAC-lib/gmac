#ifndef GMAC_API_CUDA_HPE_MODE_IMPL_H_
#define GMAC_API_CUDA_HPE_MODE_IMPL_H_

#include "core/IOBuffer.h"

#include "Context.h"

namespace __impl { namespace cuda { namespace hpe {

inline
void Mode::switchIn()
{
#ifdef USE_MULTI_CONTEXT
    getAccelerator().setCUcontext(&cudaCtx_);
#endif
}

inline
void Mode::switchOut()
{
#ifdef USE_MULTI_CONTEXT
    getAccelerator().setCUcontext(NULL);
#endif
}

inline
gmacError_t Mode::launch(gmac_kernel_id_t id, core::hpe::KernelLaunch *&kernel)
{
    KernelMap::iterator i = kernels_.find(id);
    if(i == kernels_.end()) return gmacErrorInvalidValue;

    Kernel * k = dynamic_cast<Kernel *>(i->second);
    switchIn();
    kernel = &(getCUDAContext().launch(*k));
    switchOut();

    return gmacSuccess;
}

inline gmacError_t
Mode::execute(core::hpe::KernelLaunch & launch)
{
    gmacError_t ret = prepareForCall();
    if(ret == gmacSuccess) {
        switchIn();
        ret = getAccelerator().execute(dynamic_cast<KernelLaunch &>(launch));
        switchOut();
    }
    return ret;
}

inline
gmacError_t Mode::call(dim3 Dg, dim3 Db, size_t shared, cudaStream_t tokens)
{
    switchIn();
    Context &ctx = getCUDAContext();
    gmacError_t ret = ctx.call(Dg, Db, shared, tokens);
    switchOut();
    return ret;
}

inline
gmacError_t Mode::argument(const void *arg, size_t size, off_t offset)
{
    switchIn();
    Context &ctx = getCUDAContext();
    gmacError_t ret = ctx.argument(arg, size, offset);
    switchOut();
    return ret;
}

inline Accelerator &
Mode::getAccelerator() const
{
    return *static_cast<Accelerator *>(acc_);
}

inline gmacError_t
Mode::eventTime(uint64_t &t, CUevent start, CUevent end)
{
    /* There is no switch-in because we must already be inside the mode */
    gmacError_t ret = getAccelerator().timeCUevents(t, start, end);
    return ret;
}

}}}

#endif
