#ifndef GMAC_UTIL_UNIQUE_IMPL_H_
#define GMAC_UTIL_UNIQUE_IMPL_H_

#include <cstdio>

namespace __impl { namespace util {

template <typename T>
Atomic Unique<T>::Count_ = 0;

#ifdef DEBUG
template <typename T>
Atomic UniqueDebug<T>::Count_ = 0;
#endif

template <typename T>
inline
Unique<T>::Unique()
{
    id_ = unsigned(AtomicInc(Count_)) - 1;
}

template <typename T>
inline
unsigned
Unique<T>::getId() const
{
    return id_;
}

#ifdef DEBUG
template <typename T>
inline
UniqueDebug<T>::UniqueDebug()
{
    id_ = unsigned(AtomicInc(Count_)) - 1;
}

template <typename T>
inline
unsigned
UniqueDebug<T>::getDebugId() const
{
    return id_;
}
#endif



}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
