#ifndef GMAC_TRACE_PARAVER_WINDOWS_LOCK_IMPL_H_
#define GMAC_TRACE_PARAVER_WINDOWS_LOCK_IMPL_H_

#include <windows.h>

namespace __impl { namespace trace { namespace paraver {

inline
SpinLock::SpinLock() :
    spinlock_(0)
{
}

inline
SpinLock::~SpinLock()
{
}

inline void
SpinLock::lock() const
{
    while (InterlockedExchange(&spinlock_, 1) == 1);
}

inline void
SpinLock::unlock() const
{
    InterlockedExchange(&spinlock_, 0); 
}

inline
Lock::Lock()
{
    InitializeCriticalSection(&mutex_);
}

inline
Lock::~Lock()
{
    DeleteCriticalSection(&mutex_);
}

inline void
Lock::lock() const
{
    EnterCriticalSection(&mutex_);
}

inline void
Lock::unlock() const
{
    LeaveCriticalSection(&mutex_);
}

inline
RWLock::RWLock() :
	owner_(0)
{
    InitializeSRWLock(&lock_);
}

inline
RWLock::~RWLock()
{
}

inline void
RWLock::lockRead() const
{
    AcquireSRWLockShared(&lock_);
}

inline void
RWLock::lockWrite() const
{
    AcquireSRWLockExclusive(&lock_);
	owner_ = GetCurrentThreadId();
}

inline void
RWLock::unlock() const
{
    if(owner_ == 0) ReleaseSRWLockShared(&lock_);
	else {
		owner_ = 0;
		ReleaseSRWLockExclusive(&lock_);
	}
}


}}}

#endif
