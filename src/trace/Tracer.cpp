#if defined(USE_TRACE)
#include "trace/Tracer.h"

namespace __impl { namespace trace {
Tracer *tracer = NULL;

Atomic threads_;
util::Private<int32_t> tid_;

CONSTRUCTOR(init);
static void init()
{
#if defined(USE_TRACE)
	util::Private<int32_t>::init(tid_);
	InitApiTracer();
    SetThreadState(Running);
#endif
}

DESTRUCTOR(fini);
static void fini()
{
#if defined(USE_TRACE)
	FiniApiTracer();
#endif
}


}}
#endif
