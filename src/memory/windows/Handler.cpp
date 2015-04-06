#include <csignal>
#include <cerrno>

#include "config/common.h"
#include "core/Process.h"
#include "memory/Handler.h"
#include "memory/Manager.h"
#include "trace/Tracer.h"

namespace __impl { namespace memory {

unsigned Handler::Count_ = 0;

static core::Process *Process_ = NULL;
static Manager *Manager_ = NULL;

static LONG CALLBACK segvHandler(EXCEPTION_POINTERS *ex)
{
    /* Check that we are getting an access violation exception */
    if(ex->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION)
        return EXCEPTION_CONTINUE_SEARCH;

    if(Process_ == NULL || Manager_ == NULL) return EXCEPTION_CONTINUE_SEARCH;

    Handler::Entry();
    trace::EnterCurrentFunction();

    bool writeAccess = false;
    if(ex->ExceptionRecord->ExceptionInformation[0] == 1) writeAccess = true;
    else if(ex->ExceptionRecord->ExceptionInformation[0] != 0) { Handler::Exit(); return EXCEPTION_CONTINUE_SEARCH; }

    void *addr = (void *)ex->ExceptionRecord->ExceptionInformation[1];

    if(writeAccess == false) TRACE(GLOBAL, "Read SIGSEGV for %p", addr);
    else TRACE(GLOBAL, "Write SIGSEGV for %p", addr);

    bool resolved = false;
    core::Mode *mode = Process_->owner((const hostptr_t)addr);
    if(mode != NULL) {
        if(!writeAccess) resolved = Manager_->signalRead(*mode, (hostptr_t)addr);
        else             resolved = Manager_->signalWrite(*mode, (hostptr_t)addr);
    }

    if(resolved == false) {
        fprintf(stderr, "Uoops! I could not find a mapping for %p. I will abort the execution\n", addr);
        abort();
        Handler::Exit();
        return EXCEPTION_CONTINUE_SEARCH;
    }

    trace::ExitCurrentFunction();
    Handler::Exit();

    return EXCEPTION_CONTINUE_EXECUTION;
}

void Handler::setHandler()
{
    AddVectoredExceptionHandler(1, segvHandler);

    Handler_ = this;
    TRACE(GLOBAL, "New signal handler programmed");
}

void Handler::restoreHandler()
{
        RemoveVectoredExceptionHandler(segvHandler);

        Handler_ = NULL;
        TRACE(GLOBAL, "Old signal handler restored");
}

void Handler::setProcess(core::Process &proc)
{
    Process_ = &proc;
}

void Handler::setManager(Manager &manager)
{
    Manager_ = &manager;
}



}}
