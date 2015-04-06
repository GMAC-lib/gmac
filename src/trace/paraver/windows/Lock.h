/* Copyright (c) 2011 University of Illinois
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

#ifndef GMAC_TRACE_PARAVER_WINDOWS_LOCK_H_
#define GMAC_TRACE_PARAVER_WINDOWS_LOCK_H_

#include <windows.h>

namespace __impl { namespace trace { namespace paraver {

class GMAC_LOCAL SpinLock {
protected:
    //! Spin lock value
	mutable long spinlock_;
public:
    //! Default constructor
	SpinLock();

    //! Default destructor
	~SpinLock();

    //! Get the lock
	void lock() const;

    //! Release the lock
	void unlock() const;
};


//! A Mutex lock
class GMAC_LOCAL Lock {
protected:
    //! Mutex holding the lock
	mutable CRITICAL_SECTION mutex_;
public:
    //! Default constructor
	Lock();
    //! Default destructor
	~Lock();

    //! Get the lock
	void lock() const;

    //! Release the lock
	void unlock() const;
};

//! A Read/Write lock
class GMAC_LOCAL RWLock {
protected:
    //! Read/Write lock
	mutable SRWLOCK lock_;

    //! Thread owning the lock
	mutable DWORD owner_;
public:
    //! Default constructor
	RWLock();
    
    //! Default destructor
	~RWLock();

    //! Get shared access to the lock
	void lockRead() const;

    //! Get exclusive access to the lock
	void lockWrite() const;

    //! Release the lock
	void unlock() const;
};

} } }

#include "Lock-impl.h"

#endif
