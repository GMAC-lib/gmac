#ifndef GMAC_UTIL_WINDOWS_LOCK_IMPL_H_
#define GMAC_UTIL_WINDOWS_LOCK_IMPL_H_

#include <windows.h>

namespace __impl { namespace util {

inline void
SpinLock::lock() const
{
    enter();
    while (InterlockedExchange(&spinlock_, 1) == 1);
    locked();
}

inline void
SpinLock::unlock() const
{
    exit();
    InterlockedExchange(&spinlock_, 0); 
}

inline void
Lock::lock() const
{
    enter();
    EnterCriticalSection(&mutex_);
    locked();
}

inline void
Lock::unlock() const
{
    exit();
    LeaveCriticalSection(&mutex_);
}

inline void
RWLock::lockRead() const
{
    enter();
    AcquireSRWLockShared(&lock_);
    done();
}

inline void
RWLock::lockWrite() const
{
    enter();
    AcquireSRWLockExclusive(&lock_);
	owner_ = GetCurrentThreadId();
    locked();
}

inline void
RWLock::unlock() const
{
    exit();
    if(owner_ == 0) ReleaseSRWLockShared(&lock_);
	else {
		owner_ = 0;
		ReleaseSRWLockExclusive(&lock_);
	}
}

}}

#endif
