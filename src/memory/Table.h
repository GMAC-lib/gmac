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

#ifndef GMAC_MEMORY_TABLE_H_
#define GMAC_MEMORY_TABLE_H_

#include <cstdlib>
#include <stdint.h>
#include <cmath>

#include "config/config.h"
#include "config/common.h"
#include "gmac/paraver.h"

#include "util/Logger.h"

// Compiler check to ensure that defines set by configure script
// are consistent

namespace __impl { namespace memory  { namespace vm {

typedef unsigned long addr_t;


template<typename T>
class GMAC_LOCAL Table : public __impl::util::Logger {
protected:
	static const size_t defaultSize = 512;
	size_t nEntries;

	static const addr_t Present = 0x01;
	static const addr_t Dirty   = 0x02;
	static const addr_t Mask    = ~0x03;

	T **table;

	T *entry(size_t n) const;

public:
	Table(size_t nEntries = defaultSize);
	virtual ~Table();

	bool present(size_t n) const;

	void create(size_t n, size_t size = defaultSize);
	void insert(size_t n, void *addr);
	void remove(size_t n);

	T &get(size_t n) const;
	T *value(size_t n) const;

	size_t size() const;
};

}}}

#include "Table.ipp"


#endif
