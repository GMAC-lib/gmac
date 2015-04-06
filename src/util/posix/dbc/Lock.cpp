#ifdef USE_DBC

#include "Lock.h"

namespace __dbc { namespace util {

#if !defined(__APPLE__)
SpinLock::SpinLock(const char *name) :
    __impl::util::SpinLock(name),
    locked_(false),
    owner_(0)
{
    pthread_mutex_init(&internal_, NULL);
}

SpinLock::~SpinLock()
{
    pthread_mutex_destroy(&internal_);
}

void SpinLock::lock() const
{
    pthread_mutex_lock(&internal_);
    REQUIRES(owner_ != pthread_self());
    pthread_mutex_unlock(&internal_);

    __impl::util::SpinLock::lock();

    pthread_mutex_lock(&internal_);
    ENSURES(owner_ == 0);
    ENSURES(locked_ == false);
    locked_ = true;
    owner_ = pthread_self();
    pthread_mutex_unlock(&internal_);
}

void SpinLock::unlock() const
{
    pthread_mutex_lock(&internal_);
    REQUIRES(locked_ == true);
    REQUIRES(owner_ == pthread_self());
    owner_ = 0;
    locked_ = false;

    __impl::util::SpinLock::unlock();

    pthread_mutex_unlock(&internal_);
}

#endif

Lock::Lock(const char *name) :
    __impl::util::Lock(name),
    locked_(false),
    owner_(0)
{
    pthread_mutex_init(&internal_, NULL);
}

Lock::~Lock()
{
    pthread_mutex_destroy(&internal_);
}

void Lock::lock() const
{
    pthread_mutex_lock(&internal_);
    REQUIRES(owner_ != pthread_self());
    pthread_mutex_unlock(&internal_);

    __impl::util::Lock::lock();

    pthread_mutex_lock(&internal_);
    ENSURES(owner_ == 0);
    ENSURES(locked_ == false);
    locked_ = true;
    owner_ = pthread_self();
    pthread_mutex_unlock(&internal_);
}

void Lock::unlock() const
{
    pthread_mutex_lock(&internal_);
    REQUIRES(locked_ == true);
    REQUIRES(owner_ == pthread_self());
    owner_ = 0;
    locked_ = false;

    __impl::util::Lock::unlock();

    pthread_mutex_unlock(&internal_);
}

RWLock::RWLock(const char *name) :
    __impl::util::RWLock(name),
    state_(Idle),
    writer_(0)
{
    ENSURES(pthread_mutex_init(&internal_, NULL) == 0);
}

RWLock::~RWLock()
{
    ENSURES(pthread_mutex_destroy(&internal_) == 0);
    readers_.clear();
}

void RWLock::lockRead() const
{
    pthread_mutex_lock(&internal_);
    REQUIRES(readers_.find(pthread_self()) == readers_.end() &&
             writer_ != pthread_self());
    pthread_mutex_unlock(&internal_);

    __impl::util::RWLock::lockRead();

    pthread_mutex_lock(&internal_);
    ENSURES(state_ == Idle || state_ == Read);
    state_ = Read;
    readers_.insert(pthread_self());
    pthread_mutex_unlock(&internal_);
}

void RWLock::lockWrite() const
{
    pthread_mutex_lock(&internal_);
    REQUIRES(readers_.find(pthread_self()) == readers_.end());
    REQUIRES(writer_ != pthread_self());
    pthread_mutex_unlock(&internal_);

    __impl::util::RWLock::lockWrite();

    pthread_mutex_lock(&internal_);
    ENSURES(readers_.empty() == true);
    ENSURES(state_ == Idle);
    state_ = Write;
    writer_ = pthread_self();
    pthread_mutex_unlock(&internal_);
}

void RWLock::unlock() const
{
    pthread_mutex_lock(&internal_);
    if(writer_ == pthread_self()) {
        REQUIRES(readers_.empty() == true);
        REQUIRES(state_ == Write);
        state_ = Idle;
        writer_ = 0;
    }
    else {
        REQUIRES(readers_.erase(pthread_self()) == 1);
        REQUIRES(state_ == Read);
        if(readers_.empty() == true) state_ = Idle;
    }

    __impl::util::RWLock::unlock();

    pthread_mutex_unlock(&internal_);
}

}}

#endif
