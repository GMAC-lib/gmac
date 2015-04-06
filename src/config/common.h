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


#ifndef GMAC_CONFIG_COMMON_H_
#define GMAC_CONFIG_COMMON_H_

#include "config/config.h"
#include "config/dbc/types.h"
#include "include/gmac/types.h"

#if defined(__GNUC__)
#include <stdint.h>
typedef size_t ptroff_t;

typedef unsigned long long_t;
#elif defined(_MSC_VER)
typedef unsigned __int8 uint8_t;
typedef signed __int8 int8_t;
typedef unsigned __int16 uint16_t;
typedef signed __int16 int16_t;
typedef unsigned __int32 uint32_t;
typedef signed __int32 int32_t;
typedef unsigned __int64 uint64_t;
typedef signed __int64 int64_t;
typedef signed __int64 ssize_t;
typedef int ptroff_t;

typedef ULONG_PTR long_t;
#endif

#ifndef _MSC_VER
#define UNREFERENCED_PARAMETER(a)
#endif

typedef uint8_t * hostptr_t;
#define NIL ((void *) 0)

#ifdef DEBUG
#define DEBUG_PARAM_DECLARATION(t,v) t v
#define DEBUG_PARAM(v) v
#else
#define DEBUG_PARAM_DECLARATION(t,v)
#define DEBUG_PARAM(v)
#endif

#ifdef USE_CUDA
#include "cuda/common.h"
#include "include/gmac/cuda_types.h"
#else
#ifdef USE_OPENCL
#include "opencl/common.h"
#include "include/gmac/opencl_types.h"
#else
#error "No programming model back-end specified"
#endif
#endif


namespace __impl {
    namespace cuda {}
    namespace core {}
    namespace util {}
    namespace memory {
        namespace protocol {}
    }
    namespace trace {}
}

#ifdef USE_DBC
namespace __dbc {
    namespace cuda {}
    namespace core {
        // Singleton classes need to be predeclared
        namespace hpe {
            class Process;
            class Mode;
        }
    }
    namespace util {}
    namespace memory {
        // Singleton classes need to be predeclared
        class Manager;
        namespace protocol {}
    }
    namespace trace = __impl::trace;
}
#endif
#ifdef USE_DBC
namespace gmac = __dbc;

#define DBC_FORCE_TEST(c) virtual void __dbcForceTest(c &o) = 0;
#define DBC_TESTED(c)             void __dbcForceTest(c &) {}
#else
namespace gmac = __impl;

#define DBC_FORCE_TEST(c)
#define DBC_TESTED(c)
#endif

/* Generic helper definitions for shared library support */
#if defined _WIN32 || defined __CYGWIN__
struct __constructor { 
	__constructor(void (*fn)(void)) { fn(); }
};
#  define CONSTRUCTOR(fn) void fn(); static __constructor __constructor_##fn(fn)
struct __destructor { 
	void (*fn_)(void);
	__destructor(void (*fn)(void)) : fn_(fn) { };
	~__destructor() { fn_(); }
};
#  define DESTRUCTOR(fn) void fn(); static __destructor __destructor_##fn(fn)
#else
#  define CONSTRUCTOR(fn) static void fn(void) __attribute__((constructor))
#  define DESTRUCTOR(fn)  static void fn(void) __attribute__((destructor))
#endif


#include "include/gmac/visibility.h"
#include "config/common.h"

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
