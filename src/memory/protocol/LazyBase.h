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

#ifndef GMAC_MEMORY_PROTOCOL_LAZYBASE_H_
#define GMAC_MEMORY_PROTOCOL_LAZYBASE_H_

#include "config/common.h"
#include "include/gmac/types.h"

#include "memory/Handler.h"
#include "memory/Protocol.h"
#include "util/Lock.h"

#include "common/BlockList.h"
#include "lazy/BlockState.h"

namespace __impl {

namespace core {
    class IOBuffer;
    class Mode;
}

namespace memory {
class Object;

template <typename State> class StateBlock;

namespace protocol {
/**
 * A lazy memory coherence protocol.
 *
 * This protocol eagerly transfer data from host to accelerator memory if the user
 * sets up a limit, otherwise data is transferred when the use requests a
 * release operation. Data is transferred from accelerator memory to host memory
 * lazily, whenever it is needed by the application
 */
class GMAC_LOCAL LazyBase : public Protocol, Handler, gmac::util::Lock {
    DBC_FORCE_TEST(LazyBase)

protected:
    /** Return the state corresponding to a memory protection
     *ock
     * \param prot Memory protection
     * \return Protocol state
     */
    lazy::State state(GmacProtection prot) const;

    /// Uses eager update
    bool eager_;

    /// Maximum number of blocks in dirty state
    size_t limit_;

    /// Dirty block list. List of all memory blocks in Dirty state
    BlockList dbl_;

    /// Add a new block to the Dirty Block List
    void addDirty(lazy::Block &block);

    /** Default constructor
     *
     * \param eager Tells if protocol uses eager update
     */
    explicit LazyBase(bool eager);

    /// Default destructor
    virtual ~LazyBase();

public:
    // Protocol Interface
    void deleteObject(Object &obj);

    bool needUpdate(const Block &block) const;

    TESTABLE gmacError_t signalRead(Block &block, hostptr_t addr);

    TESTABLE gmacError_t signalWrite(Block &block, hostptr_t addr);

    TESTABLE gmacError_t acquire(Block &block, GmacProtection &prot);

    TESTABLE gmacError_t release(Block &block);

#ifdef USE_VM
    gmacError_t acquireWithBitmap(Block &block);
#endif

    TESTABLE gmacError_t releaseAll();
    gmacError_t releasedAll();

    gmacError_t mapToAccelerator(Block &block);

    gmacError_t unmapFromAccelerator(Block &block);

    gmacError_t deleteBlock(Block &block);

    TESTABLE gmacError_t toHost(Block &block);

#if 0
    gmacError_t toAccelerator(Block &block);
#endif

    TESTABLE gmacError_t copyToBuffer(Block &block, core::IOBuffer &buffer, size_t size,
                                      size_t bufferOffset, size_t blockOffset);

    TESTABLE gmacError_t copyFromBuffer(Block &block, core::IOBuffer &buffer, size_t size,
                                        size_t bufferOffset, size_t blockOffset);

    TESTABLE gmacError_t memset(const Block &block, int v, size_t size,
                                size_t blockOffset);

    TESTABLE gmacError_t flushDirty();

    //bool isInAccelerator(Block &block);
    TESTABLE gmacError_t copyBlockToBlock(Block &d, size_t dstOffset, Block &s, size_t srcOffset, size_t count);

    gmacError_t dump(Block &block, std::ostream &out, common::Statistic stat);
};

}}}

#ifdef USE_DBC
#include "dbc/LazyBase.h"
#endif

#endif
/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
