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

#ifndef GMAC_UTIL_LOCK_H_
#define GMAC_UTIL_LOCK_H_

#include "config/common.h"

#if defined(USE_TRACE_LOCKS)
#include <string>
#endif

namespace __impl { namespace util {

class GMAC_LOCAL __Lock {
protected:
#if defined(USE_TRACE_LOCKS)
    //! Signal that the lock is exclusive, e.g., due to a lock-write
    mutable bool exclusive_;
    std::string name_;
#endif
public:
    //! Default constructor
    /*!
        \param name Lock name for tracing purposes
    */
    __Lock(const char *name);

    //! The thread requests the lock
    void enter() const;
    
    //! The thread gets an exclusive lock
    void locked() const;

    //! The thread gets a shared lock
    void done() const;

    //! The thread releases a lock
    void exit() const;
};

}}

#if defined(POSIX)
#include "util/posix/Lock.h"
#elif defined(WINDOWS)
#include "util/windows/Lock.h"
#endif

#include "Lock-impl.h"


#endif
