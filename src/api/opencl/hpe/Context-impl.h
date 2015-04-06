#ifndef GMAC_API_OPENCL_HPE_CONTEXT_IMPL_H_
#define GMAC_API_OPENCL_HPE_CONTEXT_IMPL_H_

#include "Kernel.h"

#include "api/opencl/hpe/Accelerator.h"

namespace __impl { namespace opencl { namespace hpe {

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
