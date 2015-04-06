#ifndef GMAC_UTIL_WINDOWS_THREAD_IMPL_H_
#define GMAC_UTIL_WINDOWS_THREAD_IMPL_H_

#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64

namespace __impl { namespace util { 

inline THREAD_T GetThreadId()
{
	return GetCurrentThreadId();
}

inline PROCESS_T GetProcessId()
{
	return GetCurrentProcessId();
}

inline
long_t GetTimeStamp()
{
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);

    unsigned __int64 tmp = 0;
    tmp |= ft.dwHighDateTime;
    tmp <<= 32;
    tmp |= ft.dwLowDateTime;
    tmp -= DELTA_EPOCH_IN_MICROSECS;
    tmp /= 10;

    return long_t(tmp);
}

}}

#endif
