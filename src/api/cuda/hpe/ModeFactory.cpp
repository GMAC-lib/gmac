#include "api/cuda/hpe/Mode.h"
#include "api/cuda/hpe/ModeFactory.h"

namespace __impl { namespace cuda { namespace hpe {

Mode *ModeFactory::create(core::hpe::Process &proc, Accelerator &acc, __impl::core::hpe::AddressSpace &aSpace) const
{
    return new gmac::cuda::hpe::Mode(proc, acc, aSpace);
}

}}}
