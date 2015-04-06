#ifndef GMAC_UTIL_LOGGER_IMPL_H_
#define GMAC_UTIL_LOGGER_IMPL_H_

#include <sstream>

namespace __impl { namespace util {
#ifdef DEBUG

#ifdef USE_CXX0X
static
unsigned numberOfVariables(const char *_str)
{
    if (_str == NULL) return 0;

    unsigned ret = 0;

    size_t pos = 0;
    std::string str(_str);
    while ((pos = str.find('%', pos)) != std::string::npos) {
        pos++;
        ret++;
    }

    return ret;
}

template <typename ...Types>
inline
void Logger::__Trace(const char *name, const char *funcName, const char *fileName, unsigned lineNumber, THREAD_T tid, const char *fmt, Types... list)
#else // USE_CXX0X
inline
void Logger::__Trace(const char *name, const char *funcName, const char *fileName, unsigned lineNumber, THREAD_T tid, const char *fmt, ...)
#endif
{
    std::stringstream newFmt;
    newFmt << std::string("(") << tid << std::string(") ") << "[" << funcName << "] ";

    if (__impl::util::params::ParamDebugPrintFile) {
        newFmt << "{" << fileName << ":" << lineNumber << "} ";
    }

    newFmt << fmt;

    if (AtomicTestAndSet(Ready_, 0, 1) == 0) Init();

#ifdef USE_CXX0X
    if (__impl::util::params::ParamDebugUseFinalClass) {
        Log(name, "TRACE", newFmt.str().c_str(), list...);
    } else {
        Log(funcName, "TRACE", newFmt.str().c_str(), list...);
    }
#else
    va_list list;
    va_start(list, fmt);

    if (__impl::util::params::ParamDebugUseFinalClass) {
        Log(name, "TRACE", newFmt.str().c_str(), list);
    } else {
        Log(funcName, "TRACE", newFmt.str().c_str(), list);
    }

    va_end(list);

#endif
}

#ifdef USE_CXX0X
template <typename ...Types>
inline
void Logger::__Assertion(bool c, const char *cStr, const char *fmt, Types... list)
#else
inline
void Logger::__Assertion(bool c, const char *cStr, const char *fmt, ...)
#endif
{
    if(c == true ) return;
#ifdef USE_CXX0X
    Print(cStr, fmt, list...);
#else
    va_list list;
    va_start(list, fmt);
    Print(cStr, fmt, list);
    va_end(list);
#endif
    abort();
}

#ifdef USE_CXX0X
template <typename ...Types>
inline
void Logger::Log(const char *name, const char *tag, const char *fmt, Types ...list)
#else
inline
void Logger::Log(const char *name, const char *tag, const char *fmt, va_list list)
#endif
{
    if(Check(name) == false) return;

#ifdef USE_CXX0X
    Print(tag, fmt, list...);
#else
    Print(tag, fmt, list);
#endif
}


#endif // DEBUG

#ifdef USE_CXX0X
template <typename ...Types>
inline
void Logger::Print(const char *tag, const char *fmt, Types ...list)
#else
inline
void Logger::Print(const char *tag, const char *fmt, va_list list)
#endif
{
    if(AtomicTestAndSet(Ready_, 0, 1) == 0) Init();
    char *buffer = Buffer_.get();
    if (buffer == NULL) {
        buffer = new char[BufferSize_];
		Buffer_.set(buffer);
	}

#ifdef DEBUG
#ifdef USE_CXX0X
    if (numberOfVariables(fmt) != sizeof...(Types)) abort();
#endif
#endif

#ifdef USE_CXX0X
	snprintf(buffer, BufferSize_, fmt, list...);
#else
	VSNPRINTF(buffer, BufferSize_, fmt, list);
#endif
	fprintf(stderr,"%s: %s\n", tag, buffer);
}


#ifdef USE_CXX0X
inline
void Logger::Print(const char *tag, const char *fmt)
{
    if(AtomicTestAndSet(Ready_, 0, 1) == 0) Init();
	if (Buffer_.get() == NULL) {
		Buffer_.set(new char[BufferSize_]);
	}
	
	fprintf(stderr,"%s: %s\n", tag, Buffer_.get());
}
#endif

inline void Logger::__Message(const char *fmt, ...)
{
    va_list list;
    va_start(list, fmt);
	VFPRINTF(stdout, fmt, list);
    va_end(list);
}


inline void Logger::__Warning(const char *fmt, ...)
{
    va_list list;
    va_start(list, fmt);
	VFPRINTF(stderr, fmt, list);
    va_end(list);
}


#ifdef USE_CXX0X
template <typename ...Types>
inline
void Logger::__Fatal(const char *fmt, Types ...list)
#else
inline
void Logger::__Fatal(const char *fmt, ...)
#endif
{
#ifdef USE_CXX0X
    Print("FATAL", fmt, list...);
#else
    va_list list;
    va_start(list, fmt);
    Print("FATAL", fmt, list);
    va_end(list);
#endif
    abort();
}

#ifdef USE_CXX0X
template <typename ...Types>
inline
void Logger::__CFatal(bool c, const char *cStr, const char *fmt, Types... list)
#else
inline
void Logger::__CFatal(bool c, const char *cStr, const char *fmt, ...)
#endif
{
    if(c == true) return;
#ifdef USE_CXX0X
    Print(cStr, fmt, list...);
#else
    va_list list;
    va_start(list, fmt);
    Print(cStr, fmt, list);
    va_end(list);
#endif
    abort();
}

}}

#endif
