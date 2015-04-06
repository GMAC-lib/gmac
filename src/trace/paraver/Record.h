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

#ifndef GMAC_TRACE_PARAVER_RECORD_H_
#define GMAC_TRACE_PARAVER_RECORD_H_

#include <list>
#include <string>
#include <iostream>
#include <fstream>

#include "config/common.h"

#include "StreamOut.h"

namespace __impl { namespace trace { namespace paraver {

class Thread;
class State;
class Event;
class Communication;

class GMAC_LOCAL Record {
protected:
	typedef enum {
		STATE=1,
		EVENT=2,
		COMM=3,
		LAST
	} Type;
public:
	virtual uint64_t getTime() const = 0;
	virtual uint64_t getEndTime() const = 0;
	virtual int getType() const = 0;
	virtual uint32_t getId() const = 0;
	virtual void write(StreamOut &of) const = 0;

	static void end(StreamOut &of);
	static Record *read(std::ifstream &in);
	friend StreamOut & operator<<(StreamOut &os, const Record &record);
    friend StreamOut & operator<<(StreamOut &os, const State &state);
    friend StreamOut & operator<<(StreamOut &os, const Event &event);
    friend StreamOut & operator<<(StreamOut &os, const Communication &comm);
};

class GMAC_LOCAL RecordPredicate {
public:
	bool operator()(const Record *a, const Record *b);
};

class GMAC_LOCAL RecordId {
protected:
	int32_t task_, app_, thread_;
public:
	RecordId(int32_t task, int32_t app, int32_t thread);
	RecordId(std::ifstream &in);
	
	void write(StreamOut &of) const;

	friend StreamOut & operator<<(StreamOut &os, const RecordId &id);
};

class GMAC_LOCAL State : public Record {
public:
	static const uint32_t None = 0;
	static const uint32_t Running = 1;
private:
	RecordId id_;
	uint64_t start_;
	uint64_t end_;
	uint32_t state_;
public:
	State(Thread *thread);
	State(std::ifstream &in);

	int getType() const;
	uint64_t getTime() const;
	uint64_t getEndTime() const;
	uint32_t getId() const;

	void start(uint32_t state, uint64_t start);
	void restart(uint64_t start);
	void end(uint64_t end);

	void write(StreamOut &of) const;
	friend StreamOut & operator<<(StreamOut &os, const State &state);
};

class GMAC_LOCAL Event : public Record {
private:
	RecordId id_;
	uint64_t when_;
	uint64_t event_;
	int64_t value_;

public:
	Event(Thread *thread, uint64_t when, uint64_t event, int64_t value);
	Event(std::ifstream &in);

	int getType() const;
	uint64_t getTime() const;
	uint64_t getEndTime() const;
	uint32_t getId() const;

	void write(StreamOut &of) const;
	friend StreamOut & operator<<(StreamOut &os, const Event &event);

};

class GMAC_LOCAL Communication : public Record {
private:
    RecordId src_, dst_;
    uint64_t start_, end_;
    uint64_t size_;
public:
    Communication(Thread *src, Thread *dst, uint64_t start, uint64_t end, uint64_t size);
    Communication(std::ifstream &in);

    int getType() const;
    uint64_t getTime() const;
    uint64_t getEndTime() const;
    uint32_t getId() const;

    void write(StreamOut &of) const;
    friend StreamOut & operator<<(StreamOut &os, const Communication &comm);
};

} } }

#include "Record-impl.h"

#endif
