#ifndef GMAC_TRACE_PARAVER_POSIX_LOCK_IMPL_H_
#define GMAC_TRACE_PARAVER_POSIX_LOCK_IMPL_H_

namespace __impl { namespace trace { namespace paraver {

inline
SpinLock::SpinLock()
{
    pthread_spin_init(&spinlock_, PTHREAD_PROCESS_PRIVATE);
}

inline
SpinLock::~SpinLock()
{
    pthread_spin_destroy(&spinlock_);
}

inline void
SpinLock::lock() const
{
    pthread_spin_lock(&spinlock_);
}

inline void
SpinLock::unlock() const
{
    pthread_spin_unlock(&spinlock_);
}

inline
Lock::Lock() 
{
    pthread_mutex_init(&mutex_, NULL);
}

inline
Lock::~Lock()
{
    pthread_mutex_destroy(&mutex_);
}


inline void
Lock::lock() const
{
    pthread_mutex_lock(&mutex_);
}

inline void
Lock::unlock() const
{
    pthread_mutex_unlock(&mutex_);
}

inline
RWLock::RWLock()
{
    pthread_rwlock_init(&lock_, NULL);
}

inline
RWLock::~RWLock()
{
    pthread_rwlock_destroy(&lock_);
}

inline void
RWLock::lockRead() const
{
    pthread_rwlock_rdlock(&lock_);
}

inline void
RWLock::lockWrite() const
{
    pthread_rwlock_wrlock(&lock_);
}

inline void
RWLock::unlock() const
{
    pthread_rwlock_unlock(&lock_);
}




} } }

#endif
