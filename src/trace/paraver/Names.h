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

#ifndef GMAC_TRACE_PARAVER_NAMES_H_
#define GMAC_TRACE_PARAVER_NAMES_H_

#include "config/common.h"
#include "util/Logger.h"

#include <string>
#include <vector>
#include <map>

namespace __impl { namespace trace { namespace paraver {

template<typename T>
class GMAC_LOCAL Factory {
public:
	typedef std::vector<const T *> List;
private:
   Factory() {};
protected:
    static int32_t next_;
    static List *items_;

   static void init();
public:
	static T *create(const char *name);
	static bool valid();
	static const List &get();
	static void destroy();
};


class GMAC_LOCAL Name {
private:
	std::string name_;
	int32_t value_;
public:
	Name(const char *name, int32_t value);

	std::string getName() const;
	int32_t getValue() const;
};


class GMAC_LOCAL StateName : public Name {
public:
	typedef std::vector<const StateName *> List;

	StateName(const char *name, int32_t value);
   friend class Factory<StateName>;
};


class GMAC_LOCAL EventName : public Name {
public:
	typedef std::vector<const EventName *> List;
	typedef std::map<uint32_t, std::string> TypeTable;
private:
	EventName(const char *name, int32_t value);
   friend class Factory<EventName>;
protected:
	TypeTable types_;
public:
	void registerType(uint32_t value, std::string type);
	const TypeTable &getTypes() const;
};



} } }

#include "Names-impl.h"

#endif
