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

#ifndef GMAC_UTIL_POSIX_LOCK_H_
#define GMAC_UTIL_POSIX_LOCK_H_

#include <pthread.h>

#include <string>
#include <iostream>
#include <map>

#include "config/config.h"
#include "config/dbc/types.h"
#include "util/Lock.h"


namespace __impl { namespace util {

#if defined(__APPLE__)
class Lock;
typedef Lock SpinLock;
#else
class GMAC_API SpinLock : public __impl::util::__Lock {
    DBC_FORCE_TEST(SpinLock)
protected:
	mutable pthread_spinlock_t spinlock_;
public:
	SpinLock(const char *name);
	VIRTUAL ~SpinLock();

protected:
	TESTABLE void lock() const;
	TESTABLE void unlock() const;
};
#endif

class GMAC_API Lock : public __impl::util::__Lock {
    DBC_FORCE_TEST(Lock)

protected:
	mutable pthread_mutex_t mutex_;
public:
	Lock(const char *name);
	VIRTUAL ~Lock();

protected:
	TESTABLE void lock() const;
	TESTABLE void unlock() const;
};

class GMAC_API RWLock : public __impl::util::__Lock {
    DBC_FORCE_TEST(RWLock)

protected:
	mutable pthread_rwlock_t lock_;
    bool write_;
public:
	RWLock(const char *name);
	VIRTUAL ~RWLock();

protected:
	TESTABLE void lockRead() const;
	TESTABLE void lockWrite() const;
	TESTABLE void unlock() const;
};

}}

#include "Lock-impl.h"

#ifdef USE_DBC
#include "dbc/Lock.h"
#endif

#endif
