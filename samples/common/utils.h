#ifndef GMAC_UTILS_H_
#define GMAC_UTILS_H_

#include <stdio.h>

#if (defined (_WIN32) && defined(_MSC_VER))

/* scalar types  */
typedef signed   __int8         cl_char;
typedef unsigned __int8         cl_uchar;
typedef signed   __int16        cl_short;
typedef unsigned __int16        cl_ushort;
typedef signed   __int32        cl_int;
typedef unsigned __int32        cl_uint;
typedef signed   __int64        cl_long;
typedef unsigned __int64        cl_ulong;
typedef unsigned __int16        cl_half;
#else /* !_WIN32 */
typedef int8_t    cl_char;
typedef uint8_t   cl_uchar;
typedef int16_t   cl_short;
typedef uint16_t  cl_ushort;
typedef int32_t   cl_int;
typedef uint32_t  cl_uint;
typedef int64_t   cl_long;
typedef uint64_t  cl_ulong;
typedef uint16_t  cl_half;
#endif /* !_WIN32 */
typedef float                   cl_float;
typedef double                  cl_double;

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#   include <windows.h>
    typedef DWORD thread_t;
#else
#   include <pthread.h>
    typedef pthread_t thread_t;
#endif

typedef void*(*thread_routine)(void *);
thread_t thread_create(thread_routine rtn, void *arg);
void thread_wait(thread_t id);

#if defined(__GNUC__)
#	define GETENV getenv
#elif defined(_MSC_VER)
#	define GETENV gmac_getenv
static inline const char *gmac_getenv(const char *name)
{
	static char buffer[512];
	size_t size = 0;
	if(getenv_s(&size, buffer, 512, name) != 0) return NULL;
	if(strlen(buffer) == 0) return NULL;
	return (const char *)buffer;
}
#endif

#ifdef __cplusplus
}
#endif

/* Param functions */
#ifdef __cplusplus

#include <cstdlib>

template<typename T>
void setParam(T *param, const char *str, const T def)
{
	const char *value = GETENV(str);
    T val = value != NULL? T(atoi(value)): def;
    *param = val;
}

template<>
inline
void setParam<bool>(bool *param, const char *str, const bool def)
{
	const char *value = GETENV(str);
    bool val = (value != NULL) ? bool(atoi(value) != 0? true: false): def;
    *param = val;
}
#endif

#endif
