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

#ifndef GMAC_MEMORY_OBJECT_H_
#define GMAC_MEMORY_OBJECT_H_

#include <map>

#include "config/common.h"
#include "include/gmac/types.h"

#include "util/Atomics.h"
#include "util/Lock.h"
#include "util/Reference.h"
#include "memory/Protocol.h"

namespace __impl {

namespace core {
    class Mode;
}

namespace memory {

/**
 * Base abstraction of the memory allocations managed by GMAC. Objects are
 * divided into blocks, which are the unit of coherence
 */
class GMAC_LOCAL Object :
    protected gmac::util::RWLock,
    public util::Reference {

    DBC_FORCE_TEST(Object)
protected:
#ifdef DEBUG
    static Atomic Id_;
    unsigned id_;
    std::map<protocol::common::Statistic, unsigned> dumps_;
#endif

    /// Object host memory address
    hostptr_t addr_;

    /// Object size in bytes
    size_t size_;

    typedef std::map<hostptr_t, Block *> BlockMap;
    /// Collection of blocks forming the object
    BlockMap blocks_;

    /// Tells whether the object has been released or not
    bool released_;

    /**
     * Returns the block corresponding to a given offset from the begining of the object
     *
     * \param objectOffset Offset (in bytes) from the begining of the object where the block is located
     * \param blockOffset Returns the block offset of the object offset
     * \return Constant iterator pointing to the block
     */
    BlockMap::const_iterator firstBlock(size_t objectOffset, size_t &blockOffset) const;

    /** Execute a coherence operation on all the blocks of the object
     *
     * \param op Coherence operation to be performed
     * \return Error code
     * \sa __impl::memory::Block::toHost
     * \sa __impl::memory::Block::toAccelerator
     */
    gmacError_t coherenceOp(gmacError_t (Protocol::*op)(Block &));

    template <typename T>
    gmacError_t coherenceOp(gmacError_t (Protocol::*op)(Block &, T &), T &param);

    /**
     * Execute a memory operation involving an I/O buffer on all the blocks of the object
     *
     * \param op Memory operation to be executed
     * \param buffer I/O buffer used in the memory operation
     * \param size Size (in bytes) of the memory operation
     * \param bufferOffset Offset (in bytes) from the begining of the buffer to start performing the operation
     * \param objectOffset Offset (in bytes) from the beginning of the block to start performing the operation
     * \return Error code
     * \sa __impl::memory::Block::copyToHost(core::IOBuffer &, size_t, size_t, size_t) const
     * \sa __impl::memory::Block::copyToAccelerator(core::IOBuffer &, size_t, size_t, size_t) const
     * \sa __impl::memory::Block::copyFromHost(core::IOBuffer &, size_t, size_t, size_t) const
     * \sa __impl::memory::Block::copyFromAccelerator(core::IOBuffer &, size_t, size_t, size_t) const
     */
    TESTABLE gmacError_t memoryOp(Protocol::MemoryOp op,
                                  core::IOBuffer &buffer, size_t size, size_t bufferOffset, size_t objectOffset);

    /**
     * Execute an operation on all the blocks of the object
     * \param f Operation to be performed
     * \param t Parameter to be passed
     * \param s Parameter to be passed
     * \return Error code
     * \sa __impl::memory::Block::dump
     */
    template <typename T, typename S>
    gmacError_t forEachBlock(gmacError_t (Block::*f)(T &, S), T &t, S s);

    /**
     * Default constructor
     *
     * \param addr Host memory address where the object begins
     * \param size Size (in bytes) of the memory object
     */
    Object(hostptr_t addr, size_t size);

    //! Default destructor
    virtual ~Object();

public:
#ifdef DEBUG
    unsigned getId() const;
    unsigned getDumps(protocol::common::Statistic stat);
#endif

    /**
     * Get the starting host memory address of the object
     *
     * \return Starting host memory address of the object
     */
    hostptr_t addr() const;

    /**
     * Get the ending host memory address of the object
     *
     * \return Ending host memory address of the object
     */
    hostptr_t end() const;

    /**
     * Get the offset to the beginning of the block that contains the address
     *
     * \return Offset to the beginning of the block that contains the address
     */
    TESTABLE ssize_t blockBase(size_t offset) const;

    /**
     * Get the offset to the end of the block that contains the address
     *
     * \return Offset to the end of the block that contains the address
     */
    TESTABLE size_t blockEnd(size_t offset) const;

    /**
     * Get the block size used by the object
     *
     * \return Block size used by the object
     */
    size_t blockSize() const;

    /**
     * Get the size (in bytes) of the object
     *
     * \return Size (in bytes) of the object
     */
    size_t size() const;

    /// Ensure the owner(s) invalidate memory when acquiring objects
    virtual void modifiedObject() = 0;

    /**
     * Get the accelerator memory address where a host memory address from the object is mapped
     *
     * \param current Execution mode requesting the translation
     * \param addr Host memory address within the object
     * \return Accelerator memory address within the object
     */
    virtual accptr_t acceleratorAddr(core::Mode &current, const hostptr_t addr) const = 0;

    /**
     * Get the owner of the object
     *
     * \param current Execution mode requesting the operation
     * \param addr Memory address within the object
     * \return The owner of the object
     */
    virtual core::Mode &owner(core::Mode &current, const hostptr_t addr) const = 0;

    /**
     * Add a new owner to the object
     *
     * \param owner The new owner of the mode
     * \return Wether it was possible to add the owner or not
     */
    virtual gmacError_t addOwner(core::Mode &owner) = 0;

    /**
     * Remove an owner from the object
     *
     * \param owner The owner to be removed
     */
    virtual gmacError_t removeOwner(core::Mode &owner) = 0;

    /**
     * Acquire the ownership of the object for the CPU
     * 
     * \param prot Access type of the previous execution on the accelerator
     * \return Error code
     */
    gmacError_t acquire(GmacProtection &prot);

#ifdef USE_VM
    /**
     * Acquire the ownership of the object for the CPU (VM version)
     *
     * \return Error code
     */
    gmacError_t acquireWithBitmap();
#endif

    /** Releases the ownership of the object for the CPU
     *
     * \return Error code
     */
    gmacError_t release();

    /** Releases the ownership of the object for the CPU and notifies the
     * protocol that all the blocks are released
     *
     * \return Error code
     */
    gmacError_t releaseBlocks();

    /**
     * Ensures that the object host memory contains an updated copy of the data
     *
     * \return Error code
     */
    gmacError_t toHost();

    /**
     * Ensures that the object accelerator memory contains an updated copy of the data
     *
     * \return Error code
     */
    gmacError_t toAccelerator();


    /**
     * Dump object information to a file
     *
     * \param param std::ostream to write information to
     * \param stat protocol::commo
     *   \sa __impl::memory::protcol::common::Statistitc
     * \return Error code
     */
    gmacError_t dump(std::ostream &param, protocol::common::Statistic stat);

    /**
     * Signal handler for faults caused due to memory reads
     *
     * \param addr Host memory address causing the fault
     * \return Error code
     */
    TESTABLE gmacError_t signalRead(hostptr_t addr);

    /**
     * Signal handler for faults caused due to memory writes
     *
     * \param addr Host memory address causing the fault
     * \return Error code
     */
    TESTABLE gmacError_t signalWrite(hostptr_t addr);

    /**
     * Copies the data from the object to an I/O buffer
     *
     * \param buffer I/O buffer where the data will be copied
     * \param size Size (in bytes) of the data to be copied
     * \param bufferOffset Offset (in bytes) from the begining of the I/O buffer to start copying the data to
     * \param objectOffset Offset (in bytes) from the begining og the object to start copying data from
     * \return Error code
     */
    TESTABLE gmacError_t copyToBuffer(core::IOBuffer &buffer, size_t size,
                                      size_t bufferOffset = 0, size_t objectOffset = 0);

    /**
     * Copies the data from an I/O buffer to the object
     *
     * \param buffer I/O buffer where the data will be copied from
     * \param size Size (in bytes) of the data to be copied
     * \param bufferOffset Offset (in bytes) from the begining of the I/O buffer to start copying the data from
     * \param objectOffset Offset (in bytes) from the begining og the object to start copying data to
     * \return Error code
     */
    TESTABLE gmacError_t copyFromBuffer(core::IOBuffer &buffer, size_t size,
                                        size_t bufferOffset = 0, size_t objectOffset = 0);

    /**
     * Initializes a memory range within the object to a specific value
     *
     * \param offset Offset within the object of the memory to be set
     * \param v Value to initialize the memory to
     * \param count Size (in bytes) of the memory region to be initialized
     * \return Error code
     */
    TESTABLE gmacError_t memset(size_t offset, int v, size_t count);

    /**
     * Adds the object to the coherence domain.
     *
     * This method ensures that the object host memory contains an updated copy of the
     * data, and then marks the object to not use the accelerator memory any more. After calling
     * this method the memory object will always remain in host memory
     * \return Error code
     */
    virtual gmacError_t mapToAccelerator() = 0;

    //! Removes the object to the coherence domain.
    /*!
        This method marks the object to use accelerator memory. After calling
        this method the object coherency is managed by the library
        \return Error code
    */
    virtual gmacError_t unmapFromAccelerator() = 0;

    /**
     * Copies data from host memory to an object
     * \param mode Execution mode requesting the memory copy
     * \param objOffset Offset (in bytes) from the begining of the object to
     * copy the data to
     * \param src Source host memory address
     * \param count Size (in bytes) of the data to be copied
     * \return Error code
     */
    gmacError_t memcpyToObject(core::Mode &mode,
                               size_t objOffset,
                               const hostptr_t src, size_t count);

    /** Copy data from object to object
     * \param mode Execution mode requesting the memory copy
     * \param dstObj Destination object
     * \param dstOffset Offset (in bytes) from the begining of the destination
     * object to copy data to
     * \param srcOffset Offset (in bytes) from the begining og the source
     * object to copy data from
     * \param count Size (in bytes) of the data to be copied
     * \return Error code
     */
    gmacError_t memcpyObjectToObject(core::Mode &mode,
                                     Object &dstObj, size_t dstOffset,
                                     size_t srcOffset,
                                     size_t count);

    /**
     * Copies data from an object to host memory
     * \param mode Execution mode requesing the memory copy
     * \param dst Destination object
     * \param objOffset Offset (in bytes) from the begining of the source object
     * to copy data from
     * \param count Size (in bytes) of the data to be copied
     * \return Error code
     */
    gmacError_t memcpyFromObject(core::Mode &mode,
                                 hostptr_t dst,
                                 size_t objOffset, size_t count);
};

}}

#include "Object-impl.h"

#ifdef USE_DBC
#include "memory/dbc/Object.h"
#endif

#endif
