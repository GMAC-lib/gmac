#ifndef GMAC_UTIL_POSIX_PRIVATE_IMPL_H_
#define GMAC_UTIL_POSIX_PRIVATE_IMPL_H_

#include <cassert>

#include <stdio.h>

namespace __impl { namespace util {

template <typename T>
inline
void Private<T>::init(Private &var)
{
    int ret = pthread_key_create(&var.key_, NULL);
    assert(ret == 0);
}


template <typename T>
inline
void Private<T>::set(const T *value)
{
    pthread_setspecific(key_, value);
}

template <typename T>
inline
T *Private<T>::get()
{
    T *ret = static_cast<T *>(pthread_getspecific(key_));
    return ret;
}

}}

#endif
