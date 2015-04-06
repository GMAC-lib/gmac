#ifndef GMAC_UTIL_POSIX_LOCK_IMPL_H_
#define GMAC_UTIL_POSIX_LOCK_IMPL_H_

#include <cassert>
#include <cstdio>

namespace __impl { namespace util {

#if !defined(__APPLE__)
inline void
SpinLock::lock() const
{
    enter();
    pthread_spin_lock(&spinlock_);
    locked();
}

inline void
SpinLock::unlock() const
{
    exit();
    pthread_spin_unlock(&spinlock_);
}
#endif

inline void
Lock::lock() const
{
    enter();
    pthread_mutex_lock(&mutex_);
    locked();
}

inline void
Lock::unlock() const
{
    exit();
    pthread_mutex_unlock(&mutex_);
}

inline void
RWLock::lockRead() const
{
    enter();
    pthread_rwlock_rdlock(&lock_);
    done();
}

inline void
RWLock::lockWrite() const
{
    enter();
    pthread_rwlock_wrlock(&lock_);
    locked();
}

inline void
RWLock::unlock() const
{
    exit();
    pthread_rwlock_unlock(&lock_);
}

}}

#endif
