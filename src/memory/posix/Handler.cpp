#include <csignal>
#include <cerrno>

#include "memory/Handler.h"
#include "memory/Manager.h"
#include "trace/Tracer.h"

#include "core/Process.h"

namespace __impl { namespace memory {

struct sigaction defaultAction;
unsigned Handler::Count_ = 0;

#if defined(LINUX)
int Handler::Signum_ = SIGSEGV;
#elif defined(DARWIN)
int Handler::Signum_ = SIGBUS;
#endif

static core::Process *Process_ = NULL;
static Manager *Manager_ = NULL;

static void segvHandler(int s, siginfo_t *info, void *ctx)
{
    if(Process_ == NULL || Manager_ == NULL) return defaultAction.sa_sigaction(s, info, ctx);
       
    Handler::Entry();
    trace::EnterCurrentFunction();
	mcontext_t *mCtx = &((ucontext_t *)ctx)->uc_mcontext;

#if defined(LINUX)
	unsigned long writeAccess = mCtx->gregs[REG_ERR] & 0x2;
#elif defined(DARWIN)
	unsigned long writeAccess = (*mCtx)->__es.__err & 0x2;
#else
    #error "Signal handler not support in this platform"
#endif
    hostptr_t addr = hostptr_t(info->si_addr);

	if(!writeAccess) TRACE(GLOBAL, "Read SIGSEGV for %p", addr);
	else TRACE(GLOBAL, "Write SIGSEGV for %p", addr);

	bool resolved = false;
    core::Mode *mode = Process_->owner(addr);
    if(mode != NULL) {
	    if(!writeAccess) resolved = Manager_->signalRead(*mode, addr);
    	else             resolved = Manager_->signalWrite(*mode, addr);
    }

	if(resolved == false) {
		fprintf(stderr, "Uoops! I could not find a mapping for %p. I will abort the execution\n", addr);
		abort();
		// TODO: set the signal mask and other stuff
		if(defaultAction.sa_flags & SA_SIGINFO) 
			return defaultAction.sa_sigaction(s, info, ctx);
		return defaultAction.sa_handler(s);
	}

    trace::ExitCurrentFunction();
    Handler::Exit();
}


void Handler::setHandler() 
{
	struct sigaction segvAction;
	memset(&segvAction, 0, sizeof(segvAction));
	segvAction.sa_sigaction = segvHandler;
	segvAction.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&segvAction.sa_mask);

	if(sigaction(Signum_, &segvAction, &defaultAction) < 0)
		FATAL("sigaction: %s", strerror(errno));

	Handler_ = this;
	TRACE(GLOBAL, "New signal handler programmed");
}

void Handler::restoreHandler()
{
	if(sigaction(Signum_, &defaultAction, NULL) < 0)
		FATAL("sigaction: %s", strerror(errno));

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
