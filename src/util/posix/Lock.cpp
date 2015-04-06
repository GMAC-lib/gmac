#include <string>

#include "Lock.h"

namespace __impl { namespace util {

#if !defined(__APPLE__)
SpinLock::SpinLock(const char *name) :
    __impl::util::__Lock(name)
{
    pthread_spin_init(&spinlock_, PTHREAD_PROCESS_PRIVATE);
}

SpinLock::~SpinLock()
{
    pthread_spin_destroy(&spinlock_);
}
#endif

Lock::Lock(const char *name) :
    __impl::util::__Lock(name)
{
    pthread_mutex_init(&mutex_, NULL);
}

Lock::~Lock()
{
    pthread_mutex_destroy(&mutex_);
}

RWLock::RWLock(const char *name) :
    __impl::util::__Lock(name)
{
    pthread_rwlock_init(&lock_, NULL);
}

RWLock::~RWLock()
{
    pthread_rwlock_destroy(&lock_);
}

}}
