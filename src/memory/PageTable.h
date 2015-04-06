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

#ifndef GMAC_MEMORY_PAGETABLE_H_
#define GMAC_MEMORY_PAGETABLE_H_

#include <stdint.h>

#include "config/common.h"
#include "config/config.h"
#include "gmac/paraver.h"

#include "memory/Table.h"
#include "util/Lock.h"
#include "util/Logger.h"
#include "util/Parameter.h"

namespace __impl { namespace memory {

//! Page Table 

//! Software Virtual Memory Table to keep translation from
// CPU to accelerator memory addresses
class GMAC_LOCAL PageTable : public __impl::util::Logger {
private:
	static const unsigned long dirShift = 30;
	static const unsigned long rootShift = 39;

	util::RWLock lock;

	static size_t tableShift;

	size_t pages;

	typedef vm::Table<vm::addr_t> Table;
	typedef vm::Table<Table> Directory;
	vm::Table<Directory> rootTable;

	int entry(const void *addr, unsigned long shift, size_t size) const;
	int offset(const void *addr) const;

	void update();
	void sync();
	void deleteDirectory(Directory *dir);

#ifdef USE_VM
	bool _clean;
	bool _valid;
#endif

	
public:
	PageTable();
	virtual ~PageTable();

	void insert(void *host, void *acc);
	void remove(void *host);
	const void *translate(const void *host);
	void *translate(void *host);

	size_t getPageSize() const;
	size_t getTableShift() const;
	size_t getTableSize() const;
};

#include "PageTable.ipp"

}}
#endif
