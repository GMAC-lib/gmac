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

#ifndef GMAC_MEMORY_DISTRIBUTEDBLOCK_H_
#define GMAC_MEMORY_DISTRIBUTEDBLOCK_H_

#include "config/common.h"
#include "config/config.h"

#include "include/gmac/types.h"

#include "Block.h"
#include "StateBlock.h"

namespace __impl {

namespace core {
        class Mode;
}

namespace memory {

template<typename State>
class GMAC_LOCAL DistributedBlock : public StateBlock<State> {
protected:
    typedef std::map<accptr_t, std::list<core::Mode *> > AcceleratorMap;
        AcceleratorMap acceleratorAddr_;

public:
    /**
     * Default construcutor
     *
     * \param protocol Memory coherence protocol used by the block
     * \param hostAddr Host memory address for applications to accesss the block
     * \param shadowAddr Shadow host memory mapping that is always read/write
     * \param size Size (in bytes) of the memory block
     * \param init Initial block state
     */
    DistributedBlock(Protocol &protocol, hostptr_t shadowAddr,
                     hostptr_t hostAddr, size_t size, typename State::ProtocolState init);

    /// Default destructor
    virtual ~DistributedBlock();

    /**
     * Get memory block owner
     *
     *  \return A reference to the owner mode of the memory block
     */
    core::Mode &owner(core::Mode &current) const;

    /**
     * Get memory block address at the accelerator
     *
     * \param current Execution mode requesting the operation
     * \param addr Address within the block
     * \return Accelerator memory address of the block
     */
    accptr_t acceleratorAddr(core::Mode &current, const hostptr_t addr) const;

    /**
     * Get memory block address at the accelerator
     *
     * \return Accelerator memory address of the block
     */
    accptr_t acceleratorAddr(core::Mode &current) const;

    gmacError_t toAccelerator(unsigned blockOff, size_t count);

    gmacError_t toHost(unsigned blockOff, size_t count);

    gmacError_t copyToBuffer(core::IOBuffer &buffer, size_t bufferOff,
                             size_t blockOff, size_t size, typename StateBlock<State>::Source src) const;

    gmacError_t copyFromBuffer(size_t blockOff, core::IOBuffer &buffer,
                               size_t bufferOff, size_t size, typename StateBlock<State>::Destination dst) const;

    gmacError_t copyFromBlock(size_t dstOff, StateBlock<State> &srcBlock,
                              size_t srcOff, size_t size,
                              typename StateBlock<State>::Destination dst,
                              typename StateBlock<State>::Source src) const;

    gmacError_t memset(int v, size_t size, size_t blockOffset, typename StateBlock<State>::Destination dst) const;

    void addOwner(core::Mode &owner, accptr_t addr);

    void removeOwner(core::Mode &owner);
};


}}

#include "DistributedBlock-impl.h"

#endif
