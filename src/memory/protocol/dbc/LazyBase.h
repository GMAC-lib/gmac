/* Copyright (c) 2009, 2010, 2011 University of Illinois
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

#ifndef GMAC_MEMORY_PROTOCOL_DBC_LAZYBASE_H_
#define GMAC_MEMORY_PROTOCOL_DBC_LAZYBASE_H_

#include "config/dbc/types.h"

namespace __dbc { namespace memory { namespace protocol {

class GMAC_LOCAL LazyBase :
    public __impl::memory::protocol::LazyBase,
    public virtual Contract {
    DBC_TESTED(__impl::memory::protocol::LazyBase)

protected:
    LazyBase(bool eager);
    virtual ~LazyBase();

    typedef __impl::memory::protocol::LazyBase Parent;
    typedef __impl::memory::Block BlockImpl;
    typedef __impl::memory::Object ObjectImpl;
    typedef __impl::memory::protocol::lazy::State StateImpl;
    typedef __impl::memory::protocol::lazy::Block LazyBlockImpl;
    typedef __impl::core::IOBuffer IOBufferImpl;

public:
    gmacError_t signalRead(BlockImpl &block, hostptr_t addr);
    gmacError_t signalWrite(BlockImpl &block, hostptr_t addr);

    gmacError_t acquire(BlockImpl &obj, GmacProtection &prot);
    gmacError_t release(BlockImpl &block);

    gmacError_t releaseAll();

    gmacError_t toHost(BlockImpl &block);

    gmacError_t copyToBuffer(BlockImpl &block, IOBufferImpl &buffer, size_t size,
                             size_t bufferOffset, size_t blockOffset);

    gmacError_t copyFromBuffer(BlockImpl &block, IOBufferImpl &buffer, size_t size,
                               size_t bufferOffset, size_t blockOffset);

    gmacError_t memset(const BlockImpl &block, int v, size_t size, size_t blockOffset);

    gmacError_t flushDirty();

    gmacError_t copyBlockToBlock(Block &d, size_t dstOffset, Block &s, size_t srcOffset, size_t count);
};

}}}

#endif
