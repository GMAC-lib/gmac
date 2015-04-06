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

#ifndef GMAC_MEMORY_STATEBLOCK_H_
#define GMAC_MEMORY_STATEBLOCK_H_

#include "config/common.h"
#include "config/config.h"

#include "include/gmac/types.h"

/** Description for __impl. */
namespace __impl { namespace memory {

/** Description for StateBlock. */
template <typename State>
class GMAC_LOCAL StateBlock :
    public gmac::memory::Block,
    public State {
    friend gmacError_t State::syncToAccelerator();
    friend gmacError_t State::syncToHost();

protected:
    /** Default construcutor
     *
     * \param protocol Memory coherence protocol used by the block
     * \param addr Host memory address for applications to accesss the block
     * \param shadow Shadow host memory mapping that is always read/write
     * \param size Size (in bytes) of the memory block
     * \param init Initial protocol state for the block
     */
    StateBlock(Protocol &protocol, hostptr_t addr, hostptr_t shadow, size_t size, typename State::ProtocolState init);

public:
    enum EndPoint {
        HOST,
        ACCELERATOR
    };

    typedef EndPoint Destination;
    typedef EndPoint Source;

    hostptr_t getShadow() const;

    /**
     * Copy data from the block host memory to an I/O buffer
     * \param buffer IOBuffer where the operation will be executed
     * \param bufferOff Offset (in bytes) from the starting of the I/O buffer
     * where the memory operation starts
     * \param count Size (in bytes) of the memory operation
     * \param blockOff Offset (in bytes) from the starting of the block where
     * the memory opration starts
     * \param src The physical source of the memory copy (HOST or ACCELERATOR)
     * \return Error code
     * \warning This method should be only called from a Protocol class
     * \sa __impl::memory::Protocol
     */
    virtual gmacError_t copyToBuffer(core::IOBuffer &buffer, size_t bufferOff,
                                     size_t blockOff, size_t count, Source src) const = 0;

    /**
     * Copy data from the block accelerator memory to an I/O buffer
     * \param blockOff Offset (in bytes) from the starting of the block where the
     * memory opration starts
     * \param buffer IOBuffer where the operation will be executed
     * \param bufferOff Offset (in bytes) from the starting of the I/O buffer
     * where the memory operation starts
     * \param count Size (in bytes) of the memory operation
     * \param dst The physical destination of the memory copy (HOST or ACCELERATOR)
     * \return Error code
     * \sa __impl::memory::Protocol
     * \warning This method should be only called from a Protocol class
     */
    virtual gmacError_t copyFromBuffer(size_t blockOff, core::IOBuffer &buffer,
                                       size_t bufferOff, size_t count, Destination dst) const = 0;

    /**
     * Copy the data from a block to the given block
     * \param dstOff Offset in bytes within the destination block
     * \param srcBlock Reference to the source block
     * \param srcOff Offset in bytes within the source block
     * \param count Size (in bytes) of the copy
     * \param dst The physical destination of the memory copy (HOST or ACCELERATOR)
     * \param src The physical source of the memory copy (HOST or ACCELERATOR)
     * \return Error code
     * \sa __impl::memory::Protocol
     * \warning This method should be only called from a Protocol class
     */
    virtual gmacError_t copyFromBlock(size_t dstOff, StateBlock<State> &srcBlock,
                                      size_t srcOff, size_t count,
                                      Destination dst, Source src) const = 0;

    /**
     * Initializes a memory range within the block host memory to a specific value
     * \param v Value to initialize the memory to
     * \param size Size (in bytes) of the memory region to be initialized
     * \param blockOffset Offset (in bytes) from the begining of the block to perform the initialization
     * \param dst The physical destination of the memset (HOST or ACCELERATOR)
     * \return Error code
     * \warning This method should be only called from a Protocol class
     * \sa __impl::memory::Protocol
     */
    virtual gmacError_t memset(int v, size_t size, size_t blockOffset, Destination dst) const = 0;
};

}}

#include "StateBlock-impl.h"


#endif
