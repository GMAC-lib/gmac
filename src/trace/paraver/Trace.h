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

#ifndef GMAC_TRACE_PARAVER_TRACE_H
#define GMAC_TRACE_PARAVER_TRACE_H

#include "config/common.h"

#include "Element.h"
#include "Record.h"
#include "Names.h"
#include "Lock.h"

#include <vector>
#include <list>
#include <string>
#include <iostream>
#include <fstream>

#include "StreamOut.h"

namespace __impl { namespace trace { namespace paraver {

class GMAC_LOCAL TraceWriter {
protected:
    std::list<Application *> apps_;
    paraver::Lock mutex_;
    StreamOut of_;

    void processTrace(Thread *thread, uint64_t t, StateName *state);
public:
    TraceWriter(const char *fileName, uint32_t pid, uint32_t tid);
    virtual ~TraceWriter();

    void addTask(uint32_t pid);
    void addThread(uint32_t pid, uint32_t tid);

    void pushState(uint64_t t, uint32_t pid, uint32_t tid, const StateName &state);
    void pushEvent(uint64_t t, uint32_t pid, uint32_t tid, uint64_t ev, int64_t value);
    void pushEvent(uint64_t t, uint32_t pid, uint32_t tid, const EventName &event, int64_t value = 0);
    void pushCommunication(uint64_t start, uint32_t srcPid, uint32_t srcTid, uint64_t end,
         uint32_t dstPid, uint32_t dstTid, uint64_t size);

    void write(uint64_t t);
    
};

class GMAC_LOCAL TraceReader {
protected:
    std::list<Application *> apps_;
    std::list<Record *> records_;

    uint64_t endTime_;

    void buildApp(std::ifstream &in);
	friend StreamOut &operator<<(StreamOut &os, const TraceReader &trace);
public:
    TraceReader(const char *fileName);
};

} } }

#endif
