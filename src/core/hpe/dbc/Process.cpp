#ifdef USE_DBC

#include "Process.h"	

#include "core/hpe/Process.h"
#include "core/hpe/Mode.h"

namespace __dbc { namespace core { namespace hpe {

Process::Process() :
    __impl::core::hpe::Process()
{
}

Process::~Process()
{
}

void
Process::initThread()
{
   __impl::core::hpe::Process::initThread();
}

void
Process::finiThread()
{
   __impl::core::hpe::Process::finiThread();
}

__impl::core::hpe::Mode*
Process::createMode(int acc)
{
    return  __impl::core::hpe::Process::createMode(acc);
}

void
Process::removeMode(__impl::core::hpe::Mode &mode)
{
    __impl::core::hpe::Process::removeMode(mode);
}

}}}
#endif


