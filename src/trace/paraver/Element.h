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

#ifndef GMAC_TRACE_PARAVER_ELEMENT_H_
#define GMAC_TRACE_PARAVER_ELEMENT_H_

#include <cassert>

#include <map>
#include <list>
#include <string>
#include <fstream>

#include "config/common.h"

#include "Record.h"
#include "StreamOut.h"

namespace __impl { namespace trace { namespace paraver {
class GMAC_LOCAL Abstract {
protected:
	int32_t id_;
	std::string name_;
public:
	Abstract(int32_t id, std::string name);
	int32_t getId() const;
	std::string getName() const;

	virtual void end(StreamOut &of, uint64_t t) const = 0;
};

template<typename P, typename S>
class GMAC_LOCAL Element : public Abstract {
protected:
	P *parent_;
	std::map<int32_t, S *> sons_;

	P *getParent() const;
	void addSon(int32_t id, S *son);
	S *getSon(int32_t id) const;
public:
	Element(P *parent, int32_t id, std::string name);
	virtual ~Element();

	size_t size() const;
	virtual void end(StreamOut &of, uint64_t t) const;
	virtual void write(StreamOut &os) const;
};

template<typename P>
class GMAC_LOCAL Element<P, void> : public Abstract {
protected:
	P *parent_;
	inline P *getParent() const;
public:
	Element(P *parent, int32_t id, std::string name);
	virtual ~Element();

	size_t size() const;
    void end(StreamOut &of, uint64_t t) const;
	void write(StreamOut &of) const;
};

class Application;
class Task;
class GMAC_LOCAL Thread : public Element<Task, void> {
protected:
    State *current_;
    uint32_t tid_;
public:
	Thread(Task *task, int32_t id, int32_t tid);

	void start(StreamOut &of, unsigned s, uint64_t t);
	void end(StreamOut &of, uint64_t t);

	int32_t getTask() const;
	int32_t getApp() const;
    int32_t getTid() const;

	std::string print() const;

	bool ready() const;
};


class GMAC_LOCAL Task : public Element<Application, Thread> {
protected:
	int32_t threads_;
public:
	Task(Application *app, int32_t id);

	Thread *addThread(int32_t id);
	Thread *getThread(int32_t id) const;
	int32_t getApp() const;
};


class GMAC_LOCAL Application : public Element<void, Task> {
protected:
	int32_t tasks_;
public:
	Application(int32_t id, std::string name);

	Task *getTask(int32_t id) const;
	Task *addTask(int32_t id);
    
	friend StreamOut &operator<<(StreamOut &os, const Application &app);

    void close();
};


} } }

#include "Element-impl.h"
#endif
