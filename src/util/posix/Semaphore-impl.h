#ifndef GMAC_UTIL_POSIX_SEMAPHORE_IMPL_H_
#define GMAC_UTIL_POSIX_SEMAPHORE_IMPL_H_

namespace __impl { namespace util {
    
inline void
Semaphore::post()
{
    pthread_mutex_lock(&_mutex);
    _val++;
    pthread_cond_signal(&_cond);
    pthread_mutex_unlock(&_mutex);
}

inline void
Semaphore::wait()
{
    pthread_mutex_lock(&_mutex);
    _val--;
    while (_val < 0) {
        pthread_cond_wait(&_cond, &_mutex);
    }
    pthread_mutex_unlock(&_mutex);
}

}}

#endif
