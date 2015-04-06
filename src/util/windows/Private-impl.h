#ifndef GMAC_UTIL_WINDOWS_PRIVATE_IMPL_H_
#define GMAC_UTIL_WINDOWS_PRIVATE_IMPL_H_

#include "util/Logger.h"

namespace __impl { namespace util {

template<typename T>
inline Private<T>::~Private()
{
	if(key_ != TLS_OUT_OF_INDEXES)
		TlsFree(key_);
}

template <typename T>
inline
void Private<T>::init(Private &var)
{
	var.key_ = TlsAlloc();
    ASSERTION(var.key_ != TLS_OUT_OF_INDEXES);
}

template <typename T>
inline
void Private<T>::set(const void *value)
{
	TlsSetValue(key_, (LPVOID)value);
}

template <typename T>
inline
T *Private<T>::get()
{
    T* ret = static_cast<T *>(TlsGetValue(key_));
	ASSERTION(GetLastError() == ERROR_SUCCESS);
	return ret;
}

}}

#endif
