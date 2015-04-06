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

#ifndef GMAC_MEMORY_POSIX_FILEMAP_H_
#define GMAC_MEMORY_POSIX_FILEMAP_H_

#include "config/common.h"
#include "util/Logger.h"
#include "util/Lock.h"

namespace __impl { namespace memory {

class GMAC_LOCAL FileMapEntry {
protected:
    int fd_;
	hostptr_t address_;
	size_t size_;
public:
	FileMapEntry(int fd, hostptr_t address, size_t size) :
	    fd_(fd), address_(address), size_(size) {};
	virtual ~FileMapEntry() {};

	inline int fd() const { return fd_; }
	inline hostptr_t address() const { return address_; }
	inline size_t size() const { return size_; }
};

class GMAC_LOCAL FileMap :
	protected std::map<hostptr_t, FileMapEntry>,
	public gmac::util::RWLock
{
protected:
	typedef std::map<hostptr_t, FileMapEntry> Parent;
public:
	FileMap();
	virtual ~FileMap();

	bool insert(int fd, hostptr_t address, size_t size);
	bool remove(hostptr_t address);
	const FileMapEntry find(hostptr_t address) const;
};

}}

#endif
