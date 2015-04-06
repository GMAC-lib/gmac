/* Copyright (c) 2009, 2010 University of Illinois
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

#ifndef GMAC_MEMORY_DBC_OBJECT_H_
#define GMAC_MEMORY_DBC_OBJECT_H_

namespace __dbc { namespace memory {

class GMAC_LOCAL Object :
    public __impl::memory::Object,
    public virtual Contract {
    DBC_TESTED(__impl::memory::Object)

protected:
	Object(hostptr_t addr, size_t size);
    virtual ~Object();

    gmacError_t memoryOp(__impl::memory::Protocol::MemoryOp op, __impl::core::IOBuffer &buffer, size_t size, size_t bufferOffset, size_t objectOffset);
public:
    ssize_t blockBase(size_t offset) const;
    size_t blockEnd(size_t offset) const;

	gmacError_t signalRead(hostptr_t addr);
    gmacError_t signalWrite(hostptr_t addr);

    gmacError_t copyToBuffer(__impl::core::IOBuffer &buffer, size_t size, 
            size_t bufferOffset = 0, size_t objectOffset = 0);
    gmacError_t copyFromBuffer(__impl::core::IOBuffer &buffer, size_t size, 
            size_t bufferOffset = 0, size_t objectOffset = 0);

    gmacError_t memset(size_t offset, int v, size_t size);
};

}}

#endif /* OBJECT_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
