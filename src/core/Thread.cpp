#include "core/Thread.h"

namespace __impl { namespace core {

__impl::util::Private<Thread> TLS::CurrentThread_;

#ifdef DEBUG
Atomic Thread::NextTID_ = 0;
#endif

}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
