#include "core/Process.h"

#include "memory/Handler.h"

namespace __impl { namespace core {

Process::Process()
{
    memory::Handler::setProcess(*this);
}

Process::~Process()
{
}

}}
