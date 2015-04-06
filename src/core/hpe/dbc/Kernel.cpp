#ifdef USE_DBC

#include "core/hpe/Kernel.h"

namespace  __dbc {namespace core { namespace hpe {

Kernel::Kernel(const __impl::core::hpe::KernelDescriptor& k) : __impl::core::hpe::Kernel(k)
{
       REQUIRES(&k != NULL);
}

Kernel::~Kernel()
{
}

}}}
#endif 

  

