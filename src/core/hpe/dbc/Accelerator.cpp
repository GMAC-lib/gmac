#ifdef USE_DBC

#include "core/hpe/Accelerator.h"

namespace __dbc { namespace core  { namespace hpe {
    
Accelerator::Accelerator(int n) : __impl::core::hpe::Accelerator(n)
{
}

Accelerator::~Accelerator()
{
}

void Accelerator::registerMode(__impl::core::hpe::Mode &mode)
{
   REQUIRES(&mode != NULL);
  //gmacError_t ret;
    __impl::core::hpe::Accelerator::registerMode(mode);
}

void Accelerator::unregisterMode(__impl::core::hpe::Mode &mode)
{
   REQUIRES(&mode != NULL);       
  // gmacError_t ret; 
    __impl::core::hpe::Accelerator::unregisterMode(mode);
}
 
/*
// Declarations of  those method  __impl namespace  in __dbc namespace 
inline unsigned  __impl::core::Accelerator::load() const; 
inline unsigned  __impl::core::Accelerator::id() const; 
inline unsigned  __impl::core::Accelerator::busId_() const;
inline unsigned  __impl::core::Accelerator::busAccId() const;
inline bool __impl::core::Accelerator::integrated() const;
*/

}}}
#endif

