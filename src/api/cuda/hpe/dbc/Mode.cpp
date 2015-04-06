#ifdef USE_DBC
#include "api/cuda/hpe/Mode.h"

namespace __dbc { namespace cuda { namespace hpe {

Mode::Mode(__impl::core::hpe::Process &proc, __impl::cuda::hpe::Accelerator &acc, __impl::core::hpe::AddressSpace &aSpace) :
    __impl::cuda::hpe::Mode(proc, acc, aSpace)
{
}

Mode::~Mode()
{
}

}}}
#endif
/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
