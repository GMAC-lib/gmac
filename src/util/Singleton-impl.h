#ifndef GMAC_UTIL_SINGLETON_IMPL_H_
#define GMAC_UTIL_SINGLETON_IMPL_H_

#include "Singleton.h"
#include "util/Logger.h"

namespace __impl { namespace util {

template<typename T> T *Singleton<T>::Singleton_ = NULL;

template<typename T>
inline Singleton<T>::Singleton()
{
    CFATAL(Singleton_ == NULL, "Double initialization of singleton class");
    Singleton_ = static_cast<T *>(this);
}

template <typename T>
inline Singleton<T>::~Singleton()
{
}

template <typename T>
inline void Singleton<T>::destroy()
{
	ASSERTION(Singleton_ != NULL);
	delete static_cast<Singleton<T> *>(Singleton_);
	Singleton_ = NULL;
}

template <typename T>
inline T *Singleton<T>::getInstance()
{
	return Singleton_;
}



}}

#endif
