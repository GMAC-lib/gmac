#ifndef GMAC_CORE_HPE_DBC_KERNEL_H_
#define GMAC_CORE_HPE_DBC_KERNEL_H_

#include "core/hpe/Kernel.h"
#include "config/dbc/Contract.h"

namespace __dbc { namespace core { namespace hpe {

class  GMAC_LOCAL Kernel : 
    public __impl::core::hpe::Kernel,
    public virtual Contract { 
    DBC_TESTED(__impl::core::hpe::Kernel)
public:
   Kernel(const __impl::core::hpe::KernelDescriptor& k);
   virtual ~Kernel();
};
}}}

#endif  
  
