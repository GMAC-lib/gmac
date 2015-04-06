/**
 * \file pthread.cpp
 *
 * Initialization routines for pthreads
 */

#include <pthread.h>

#include "config/common.h"
#include "config/order.h"

#include "core/hpe/Process.h"

#include "hpe/init.h"

#include "trace/Tracer.h"

#include "util/loader.h"
#include "util/Lock.h"


using namespace __impl::core::hpe;

SYM(int, pthread_create__, pthread_t *__restrict, __const pthread_attr_t *, void *(*)(void *), void *);

void threadInit(void)
{
    LOAD_SYM(pthread_create__, pthread_create);
}

static void __attribute__((destructor())) gmacPthreadFini(void)
{
}

struct gmac_thread_t {
    void *(*start_routine)(void *);
    void *arg;
    bool externCall;
};

static void *gmac_pthread(void *arg)
{
    gmac::trace::StartThread("CPU");

    gmac_thread_t *gthread = (gmac_thread_t *)arg;
    bool externCall = gthread->externCall;

    // This TLS variable is necessary before entering GMAC
    if (externCall == false) {
        isRunTimeThread_.set(&privateTrue);
    } else {
        isRunTimeThread_.set(&privateFalse);
    }

    enterGmac();

    Process &proc = getProcess();
    proc.initThread();
    gmac::trace::SetThreadState(gmac::trace::Running);
    if(externCall) exitGmac();
    void *ret = gthread->start_routine(gthread->arg);
    if(externCall) enterGmac();

    // Modes and Contexts already destroyed in Process destructor
    proc.finiThread();
    free(gthread);
    gmac::trace::SetThreadState(gmac::trace::Idle);
    exitGmac();
    return ret;
}

int pthread_create(pthread_t *__restrict newthread,
                   __const pthread_attr_t *__restrict attr,
                   void *(*start_routine)(void *),
                   void *__restrict arg)
{
    int ret = 0;
    bool externCall = inGmac() == 0;
    if(externCall) enterGmac();
    TRACE(GLOBAL, "New POSIX thread");
    gmac_thread_t *gthread = (gmac_thread_t *)malloc(sizeof(gmac_thread_t));
    gthread->start_routine = start_routine;
    gthread->arg = arg;
    gthread->externCall = externCall;
    ret = pthread_create__(newthread, attr, gmac_pthread, (void *)gthread);
    if(externCall) exitGmac();
    return ret;
}
