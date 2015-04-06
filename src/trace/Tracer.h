/* Copyright (c) 2009, 2011 University of Illinois
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

#ifndef GMAC_TRACE_TRACER_H_
#define GMAC_TRACE_TRACER_H_

#include "include/gmac/types.h"
#include "config/common.h"
#include "util/Private.h"
#include "util/Atomics.h"

#include "States.h"

#if defined(__GNUC__)
#define EnterCurrentFunction() EnterFunction(__PRETTY_FUNCTION__)
#define ExitCurrentFunction()  ExitFunction(__PRETTY_FUNCTION__)
#elif defined(_MSC_VER)
#define EnterCurrentFunction() EnterFunction(__FUNCTION__)
#define ExitCurrentFunction()  ExitFunction(__FUNCTION__)
#endif


namespace __impl { namespace trace {
#if defined(USE_TRACE)
extern Atomic threads_;
static const int32_t TID_INVALID = -1;
extern util::Private<int32_t> tid_;

class GMAC_LOCAL Tracer {
protected:
    uint64_t base_;

public:
    //! Default constructor
    Tracer();

    //! Default destructor
    virtual ~Tracer();

    //! Trace the creation of a thread
    /**
        \param tid Thread ID for the created thread
        \param name Name for the created thread
        \param t Timestamp
    */
    virtual void startThread(uint64_t t, THREAD_T tid, const char *name) = 0;

    //! Trace the destruction of a thread
    /**
        \param tid Thread ID for the destroyed thread
        \param t Timestamp
    */
    virtual void endThread(uint64_t t, THREAD_T tid) = 0;

    //! Trace entering a GMAC function
    /**
        \param tid Thread ID entering the function
        \param name Name of the function
        \param t Timestamp
    */
    virtual void enterFunction(uint64_t t, THREAD_T tid, const char *name) = 0;

    //! Trace exiting from a GMAC function
    /**
        \param tid Thread existing the function
        \param name Name of the function being exited
        \param t Timestamp
    */
    virtual void exitFunction(uint64_t t, THREAD_T tid, const char *name) = 0;

#ifdef USE_TRACE_LOCKS
    //! Trace requesting a GMAC lock
    /**
        \param tid Thread ID aquiring the lock
        \param name Name of the lock
        \param t Timestamp
    */
    virtual void requestLock(uint64_t t, THREAD_T tid, const char *name) = 0;

    //! Trace acquiring a GMAC exclusive lock
    /**
        \param tid Thread ID aquiring the lock
        \param name Name of the lock being acquired
        \param t Timestamp
    */
    virtual void acquireLockExclusive(uint64_t t, THREAD_T tid, const char *name) = 0;

    //! Trace acquiring a GMAC shared lock
    /**
        \param tid Thread existing the lock
        \param name Name of the lock being acquired
        \param t Timestamp
    */
    virtual void acquireLockShared(uint64_t t, THREAD_T tid, const char *name) = 0;

    //! Trace releasing a GMAC lock
    /**
        \param tid Thread existing the lock
        \param name Name of the lock being released
        \param t Timestamp
    */
    virtual void exitLock(uint64_t t, THREAD_T tid, const char *name) = 0;
#endif

    //! Trace a change in the thread state
    /**
        \param tid Thread ID of the thread chaning its state
        \param state New thread's state
        \param t Timestamp
    */
    virtual void setThreadState(uint64_t t, THREAD_T tid, const State state) = 0;

    //! Trace a data communication between threads
    /**
        \param src ID of the thread sending the data
        \param dst ID of the thread getting the data
        \param delta Time taken to transfer the data
        \param size Size (in bytes) of the data being transferred
        \param t Timestamp
    */
    virtual void dataCommunication(uint64_t t, THREAD_T src, THREAD_T dst, uint64_t delta, size_t size) = 0;

    //! Get the current system time
    /**
        \return Time mark to be used by the traces
    */
    uint64_t timeMark() const;
};
#endif

//! Notify the creation of a new thread
/**
    \param tid Thread ID of the new thread
    \param name Name of the new thread
*/
void StartThread(THREAD_T tid, const char *name);

//! Notify the creation of a the calling thread
/**
    \param name Name of the new thread
*/
void StartThread(const char *name);

//! Notify the destruction of a thread
/**
    \param tid ID of the thread being destroyed
*/
void EndThread(THREAD_T tid);

//! Notify the destruction of the calling thread
void EndThread();

//! Nofify that a thread enters a function
/**
    \param tid ID of the thread entering the function
    \param name Name of the function
*/
void EnterFunction(THREAD_T tid, const char *name);

//! Notify that the current thread enters a function
/**
    \param name Name of the function
*/
void EnterFunction(const char *name);

//! Notifiy that a thread exits a function
/**
    \param tid ID of the thread entering the function
    \param name Name of the funcion
*/
void ExitFunction(THREAD_T tid, const char *name);

//! Notify that the current thread exists a function
/**
    \param name Name of the function
*/
void ExitFunction(const char *name);

//! Notify that the current thread requests a lock
/**
    \param name Name of the lock
*/
void RequestLock(const char *name);

//! Notify that the current thread acquires a exclusive lock
/**
    \param name Name of the lock
*/
void AcquireLockExclusive(const char *name);

//! Notify that the current thread acquires a shared lock
/**
    \param name Name of the lock
*/
void AcquireLockShared(const char *name);

//! Notify that the current thread releases a lock
/**
    \param name Name of the lock
*/
void ExitLock(const char *name);

//! Set the state of a thread
/**
    \param tid ID of the thread whose state is changing
    \param state New thread's state
*/
void SetThreadState(THREAD_T tid, const State &state);

//! Set the state of the current thread
/**
    \param state New thread's state
*/
void SetThreadState(const State &state);

//! Notify data communication between two threads
/**
    \param src ID of the thread sending the data
    \param dst ID of the thread receiving the data
    \param delta Time taken to transfer the data
    \param size Size (in bytes) of the data transferred
*/
void DataCommunication(THREAD_T src, THREAD_T dst, uint64_t delta, size_t size, uint64_t start = 0);

//! Notify a data communication started by the current thread
/**
    \param tid ID of the thread receiving the data
    \param delta Time taken to transfer the data
    \param size Size (in bytes) of the data transferred
*/
void DataCommunication(THREAD_T tid, uint64_t delta, size_t size, uint64_t start = 0);

//! Get a time mark to be used by the tracer
/**
    \param mark Reference to store the timemark
*/
void TimeMark(uint64_t &mark);

#if defined(USE_TRACE)
//! Get the thread Id for tracing purposes
/**
    \return Thread ID
*/
int32_t GetThreadId();
#endif

}}

#include "Tracer-impl.h"

#endif
