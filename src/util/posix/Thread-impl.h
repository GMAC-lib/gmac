#ifndef GMAC_UTIL_POSIX_THREAD_IMPL_H_
#define GMAC_UTIL_POSIX_THREAD_IMPL_H_

#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

namespace __impl { namespace util { 

inline THREAD_T GetThreadId()
{
	return pthread_self();
}

inline PROCESS_T GetProcessId()
{
	return getpid();
}

inline
long_t GetTimeStamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL); 

    long_t ret;

    ret = tv.tv_sec * 1000000 + tv.tv_usec;
    return ret;
}

}}

#endif
