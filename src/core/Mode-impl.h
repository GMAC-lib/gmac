#ifndef GMAC_CORE_MODE_IMPL_H_
#define GMAC_CORE_MODE_IMPL_H_

#include "config/order.h"
#include "memory/Object.h"
#include "memory/ObjectMap.h"
#include "memory/Protocol.h"
#include "trace/Tracer.h"

namespace __impl { namespace core {

inline
Mode::Mode() :
    util::Reference("Mode"),
    gmac::util::SpinLock("Mode")
{
    TRACE(LOCAL,"Creating Execution Mode %p", this);
    trace::StartThread(THREAD_T(getId()), "GPU");
    SetThreadState(THREAD_T(getId()), trace::Idle);
}

inline
Mode::~Mode()
{
    trace::EndThread(THREAD_T(getId()));
    TRACE(LOCAL,"Destroying Execution Mode %p", this);
}


} }

#endif
