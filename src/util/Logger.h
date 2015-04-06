/* Copyright (c) 2009 University of Illinois
                   Universitat Politecnica de Catalunya
                   All rights reserved.

Developed by: IMPACT Research Group / Grup de Sistemes Operatius
              University of Illinois / Universitat Politecnica de Catalunya
              http://impact.crhc.illinois.edu/
              http://gso.ac.upc.edu/

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal with the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
  1. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimers.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimers in the
     documentation and/or other materials provided with the distribution.
  3. Neither the names of IMPACT Research Group, Grup de Sistemes Operatius,
     University of Illinois, Universitat Politecnica de Catalunya, nor the
     names of its contributors may be used to endorse or promote products
     derived from this Software without specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
WITH THE SOFTWARE.  */

#ifndef GMAC_UTIL_LOGGER_H_
#define GMAC_UTIL_LOGGER_H_

#include <stdlib.h>
#include <string.h>

#include <list>
#include <string>
#include <cstdarg>
#include <cassert>
#include <typeinfo>

#include "config/common.h"
#include "config/config.h"
#include "include/gmac/types.h"
#include "util/Thread.h"
#include "util/UniquePtr.h"


#if defined(__GNUC__)
#	include <strings.h>
#	define STRTOK strtok_r
#	define VSNPRINTF vsnprintf
#   define VFPRINTF vfprintf
#	define LOCATION_STRING " in function %s [%s:%d]", __func__, __FILE__, __LINE__
#elif defined(_MSC_VER)
#	define STRTOK strtok_s
#	define VSNPRINTF(str, size, format, ap) vsnprintf_s(str, size, size - 1, format, ap)
#	define VFPRINTF(file, format, ap) vfprintf_s(file, format, ap)
#	define LOCATION_STRING " in function %s [%s:%d]", __FUNCTION__, __FILE__, __LINE__
#endif

#define SRC_ROOT "src"
inline static const char *__extract_file_name(const char *file) {
    const char *guess = strstr(file, SRC_ROOT);
    if(guess == NULL) return file;
    return guess;
}

#define GLOBAL "GMAC"

#if defined(__GNUC__) && !defined(__APPLE__)
#include <cxxabi.h>

extern __thread char nameBuffer[1024];
std::string
get_class_name(const char *mangled);

#define get_name(n) get_class_name(n).c_str()
#else
#define get_name(n) n
#endif

#define LOCAL get_name(typeid(*this).name())

#if defined(DEBUG)
#   if defined(__GNUC__)
#	    define TRACE(name, fmt, ...) __impl::util::Logger::__Trace(name, \
                                                                   __PRETTY_FUNCTION__, \
                                                                   __extract_file_name(__FILE__), __LINE__, \
                                                                   __impl::util::params::ParamDebugUseRealTID? __impl::util::GetThreadId(): __impl::core::Thread::getDebugTID(), \
                                                                   fmt, \
                                                                   ##__VA_ARGS__)
#   elif defined(_MSC_VER)
#	    define TRACE(name, fmt, ...) __impl::util::Logger::__Trace(name, \
                                                                   __FUNCTION__, \
                                                                   __extract_file_name(__FILE__), __LINE__, \
                                                                   __impl::util::params::ParamDebugUseRealTID? __impl::util::GetThreadId(): __impl::core::Thread::getDebugTID(), \
                                                                   fmt, \
                                                                   ##__VA_ARGS__)
#   endif
#   define ASSERTION(c, ...) __impl::util::Logger::__Assertion(c, "Assertion '"#c"' failed", LOCATION_STRING)
#else

static inline
void dummy_trace(...)
{
}

static inline
void dummy_assertion(bool /*b*/, ...)
{
}

#   define TRACE(...) dummy_trace(__VA_ARGS__)
#   define ASSERTION(c, ...) dummy_assertion(c, ##__VA_ARGS__)
#endif

#ifdef DEBUG
#define MESSAGE(fmt, ...) do {                                        \
                            if (__impl::util::params::ParamVerbose) { \
                                __impl::util::Logger::__Message("<GMAC> "fmt"\n", ##__VA_ARGS__); \
                                TRACE(GLOBAL, fmt, ##__VA_ARGS__);    \
                            }                                         \
                          } while (0)
#else
#define MESSAGE(fmt, ...) { if (__impl::util::params::ParamVerbose) __impl::util::Logger::__Message("<GMAC> "fmt"\n", ##__VA_ARGS__); }
#endif

#define WARNING(fmt, ...) __impl::util::Logger::__Warning("("FMT_TID")" fmt, __impl::util::GetThreadId(), ##__VA_ARGS__)
#define FATAL(fmt, ...) __impl::util::Logger::__Fatal(fmt, ##__VA_ARGS__)
#define CFATAL(c, ...) __impl::util::Logger::__CFatal(c, "Condition '"#c"' failed", LOCATION_STRING)

#include "util/Atomics.h"
#include "util/Parameter.h"
#include "util/Private.h"

namespace __impl { namespace util {

class GMAC_LOCAL Logger {
private:    
	static Atomic Ready_;
	static Private<char> Buffer_;
    static const size_t BufferSize_ = 1024;

#ifdef DEBUG
    typedef Parameter<const char *> Level;
    typedef std::list<std::string> Tags;

    static const char *DebugString_;
    static Level *Level_;
    static Tags *Tags_;

    static bool Check(const char *name);

#ifdef USE_CXX0X
    template <typename ...Types>
	static void Log(const char *name, const char *tag, const char *fmt, Types ...list);
#else // TODO: remove this as soon as the other platforms support variadic templates
	static void Log(const char *name, const char *tag, const char *fmt, va_list list);
#endif // __GNUC__

#endif // DEBUG

#ifdef USE_CXX0X
    template <typename ...Types>
    static void Print(const char *tag, const char *fmt, Types ...list);
    static void Print(const char *tag, const char *fmt);
#else // TODO: remove this as soon as the other platforms support variadic templates
    static void Print(const char *tag, const char *fmt, va_list list);
#endif // __GNUC__

public:
	static void Init();
    static void Fini();
#ifdef DEBUG
#ifdef USE_CXX0X
    template <typename ...Types>
    static void __Trace(const char *name, const char *funcName, const char *fileName, unsigned lineNumber, THREAD_T tid, const char *fmt, Types ...list);
    template <typename ...Types>
    static void __Assertion(bool c, const char * cStr, const char *fmt, Types ...list);
#else // TODO: remove this as soon as the other platforms support variadic templates
    static void __Trace(const char *name, const char *funcName, const char *fileName, unsigned lineNumber, THREAD_T tid, const char *fmt, ...);
    static void __Assertion(bool c, const char * cStr, const char *fmt, ...);
#endif
#endif // DEBUG

    static void __Message(const char *fmt, ...);
    static void __Warning(const char *fmt, ...);
#ifdef USE_CXX0X
    template <typename ...Types>
    static void __Fatal(const char *fmt, Types ...list);
    template <typename ...Types>
    static void __CFatal(bool c, const char * cStr, const char *fmt, Types ...list);
#else // TODO: remove this as soon as the other platforms support variadic templates
    static void __Fatal(const char *fmt, ...);
    static void __CFatal(bool c, const char * cStr, const char *fmt, ...);
#endif
};

}}

// Needs to be here
#include "core/Thread.h"

#include "Logger-impl.h"

#endif
