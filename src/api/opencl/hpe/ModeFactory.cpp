#include "api/opencl/hpe/Mode.h"
#include "api/opencl/hpe/ModeFactory.h"

namespace __impl { namespace opencl { namespace hpe {

Mode *ModeFactory::create(core::hpe::Process &proc, Accelerator &acc, core::hpe::AddressSpace &aSpace) const
{
    return new gmac::opencl::hpe::Mode(proc, acc, aSpace);
}

}}}
