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

#ifndef GMAC_MEMORY_BLOCK_H_
#define GMAC_MEMORY_BLOCK_H_

#include "config/common.h"
#include "config/config.h"

#include "include/gmac/types.h"
#include "memory/Protocol.h"
#include "util/Lock.h"
#include "util/Logger.h"
#include "util/Reference.h"

/** Description for __impl. */
namespace __impl {

namespace core {
    class Mode;
    class IOBuffer;
}

namespace memory {

/** Memory block
 * A memory block is a coherence unit of shared memory objects in GMAC, which are a collection of memory blocks.  Each
 * memory block has an unique host memory address, used by applications to access the shared data in the CPU code, and
 * a shadow host memory address used by GMAC to update the contents of the block. Upon creation, a memory block also
 * has one or more accelerator memory addresses, used by the application to access the data from the accelerator, and
 * owners which are those execution modes allowed to access the memory block from the accelerator. However, a memory
 * block might lose its owner and memory addresses (e.g., when the execution mode owning the memory block dies) and
 * stil be accessible from the CPU.
 * Memory block methods should only be called from GMAC objects and GMAC memory coherence protocols.
 */
class GMAC_LOCAL Block : public gmac::util::Lock,
                         public util::Reference {
    DBC_FORCE_TEST(Block)

    friend class Object;

protected:
    /** Memory coherence protocol used by the block */
    Protocol &protocol_;

    /** Block size (in bytes) */
    size_t size_;

    /** Host address where for applications to access the block. */
    hostptr_t addr_;

    /** Shadow host memory mapping that is always read/write. */
    hostptr_t shadow_;

    /**
     * Default construcutor
     * \param protocol Memory coherence protocol used by the block
     * \param addr Host memory address for applications to accesss the block
     * \param shadow Shadow host memory mapping that is always read/write
     * \param size Size (in bytes) of the memory block
     */
    Block(Protocol &protocol, hostptr_t addr, hostptr_t shadow, size_t size);

    /**
     * Default destructor
     */
    virtual ~Block();

public:

    /**
     * Host memory address where the block starts
     * \return Starting host memory address of the block
     */
    hostptr_t addr() const;

    /**
     *  Block size
     * \return Size in bytes of the memory block
     */
    size_t size() const;

protected:
    /**
     * Signal handler for faults caused due to memory reads
     * \param addr Faulting address
     * \return Error code
     */
    gmacError_t signalRead(hostptr_t addr);

    /**
     * Signal handler for faults caused due to memory writes
     * \param addr Faulting address
     * \return Error code
     */
    gmacError_t signalWrite(hostptr_t addr);

    /**
     * Ensures that the host memory has a valid and accessible copy of the data
     * \return Error code
     */
    gmacError_t toAccelerator() { return toAccelerator(0, size_); }
    virtual gmacError_t toAccelerator(unsigned blockOff, size_t count) = 0;

    /**
     * Ensures that the host memory has a valid and accessible copy of the data
     * \return Error code
     */
    gmacError_t toHost() { return toHost(0, size_); }

    /**
     * Ensures that the host memory has a valid and accessible copy of the data
     * \param blockOff Offset within the block
     * \param count Size (in bytes)
     * \return Error code
     */
    virtual gmacError_t toHost(unsigned blockOff, size_t count) = 0;

public:
    /**
     * Initializes a memory range within the block to a specific value
     * \param v Value to initialize the memory to
     * \param size Size (in bytes) of the memory region to be initialized
     * \param blockOffset Offset (in bytes) from the begining of the block to perform the initialization
     * \return Error code
     */
    TESTABLE gmacError_t memset(int v, size_t size, size_t blockOffset);

    /** Request a memory coherence operation
     *
     *  \param op Memory coherence operation to be executed
     *  \return Error code
     */
    template <typename R>
    R coherenceOp(R (Protocol::*op)(Block &));
    template <typename R, typename T>
    R coherenceOp(R (Protocol::*op)(Block &, T &), T &param);

    gmacError_t copyOp(Protocol::CopyOp op, Block &dst, size_t dstOff, size_t srcOff, size_t count);

    /**
     *  Request a memory operation over an I/O buffer
     * \param op Memory operation to be executed
     * \param buffer IOBuffer where the operation will be executed
     * \param size Size (in bytes) of the memory operation
     * \param bufferOffset Offset (in bytes) from the starting of the I/O buffer where the memory operation starts
     * \param blockOffset Offset (in bytes) from the starting of the block where the memory opration starts
     * \return Error code
     * \warning This method should be only called from a Protocol class
     * \sa copyToHost(core::IOBuffer &, size_t, size_t, size_t) const
     * \sa copyToAccelerator(core::IOBuffer &, size_t, size_t, size_t) const
     * \sa copyFromHost(core::IOBuffer &, size_t, size_t, size_t) const
     * \sa copyFromAccelerator(core::IOBuffer &, size_t, size_t, size_t) const
     * \sa __impl::memory::Protocol
     */
    TESTABLE gmacError_t memoryOp(Protocol::MemoryOp op,
            core::IOBuffer &buffer, size_t size, size_t bufferOffset, size_t blockOffset);

    /**
     * Copy data from a GMAC object to the memory block
     *
     * \param obj GMAC memory object to copy data from
     * \param size Size (in bytes) of the data to be copied
     * \param blockOffset Offset (in bytes) from the begining of the block to
     * copy the data to
     * \param objectOffset Offset (in bytes) from the begining of the object to
     * copy the data from
     * \return Error code
     */
    gmacError_t memcpyFromObject(const Object &obj, size_t size,
        size_t blockOffset = 0, size_t objectOffset = 0);

    /**
     * Get memory block owner
     * \return Owner of the memory block
     */
    virtual core::Mode &owner(core::Mode &current) const = 0;

    /**
     * Get memory block address at the accelerator
     * \return Accelerator memory address of the block
     */
    virtual accptr_t acceleratorAddr(core::Mode &current, const hostptr_t addr) const = 0;

    /**
     * Get memory block address at the accelerator
     * \return Accelerator memory address of the block
     */
    virtual accptr_t acceleratorAddr(core::Mode &current) const = 0;
 
    /**
     * Get the protocool that is managing the block
     * \return Memory protocol
     */
    Protocol &getProtocol();

    /**
     * Dump statistics about the memory block
     * \param param Stream to dump the statistics to
     * \param stat Statistic to be dumped
     * \return Error code
     */
    gmacError_t dump(std::ostream &param, protocol::common::Statistic stat);
};


}}

#include "Block-impl.h"

#ifdef USE_DBC
#include "memory/dbc/Block.h"
#endif

#endif
