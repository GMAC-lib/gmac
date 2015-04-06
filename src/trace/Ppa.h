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

#ifndef GMAC_TRACE_PPA_H_
#define GMAC_TRACE_PPA_H_

#if defined(USE_TRACE_PPA)
#include "Tracer.h"
#include "config/common.h"

#include <iostream>

namespace __impl { namespace trace {

class GMAC_LOCAL Ppa : public Tracer {
protected:
    std::string baseName_, fileName_;
    ppa::TraceWriter trace_;

    typedef std::map<std::string, int32_t > FunctionMap;
    FunctionMap functions_;
   
public:
    Ppa();
    ~Ppa();

    void startThread(uint64_t t, THREAD_T tid, const char *name);
    void endThread(uint64_t t, THREAD_T tid);

    void enterFunction(uint64_t t, THREAD_T tid, const char *name);
    void exitFunction(uint64_t t, THREAD_T tid, const char *name);

#ifdef USE_TRACE_LOCKS
    void requestLock(uint64_t t, THREAD_T tid, const char *name);
    void acquireLockExclusive(uint64_t t, THREAD_T tid, const char *name);
    void acquireLockShared(uint64_t t, THREAD_T tid, const char *name);
    void exitLock(uint64_t t, THREAD_T tid, const char *name);
#endif

    void setThreadState(uint64_t t, THREAD_T tid, const State state);
    
    void dataCommunication(uint64_t t, THREAD_T src, THREAD_T dst, uint64_t delta, size_t size);
};

} }

#endif

#endif
