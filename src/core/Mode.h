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

#ifndef GMAC_CORE_MODE_H_
#define GMAC_CORE_MODE_H_

#include "config/common.h"

#ifdef USE_VM
#include "memory/vm/Bitmap.h"
#endif
#include "util/Atomics.h"
#include "util/Lock.h"
#include "util/NonCopyable.h"
#include "util/Reference.h"
#include "util/Unique.h"

namespace __impl {

namespace memory {
class Protocol;
class Object;
class ObjectMap;
}

namespace core {

class IOBuffer;
class Process;

/**
 * A Mode represents the address space of a thread in an accelerator. Each
 * thread has one mode per accelerator type in the system
 */
class GMAC_LOCAL Mode :
    public util::Reference,
    public util::NonCopyable,
    public util::Unique<Mode>,
    public gmac::util::SpinLock {
protected:
    /**
     * Mode constructor
     */
    Mode();

    /**
     * Mode destructor
     */
    virtual ~Mode();

public:
    /** Allocate GPU-accessible host memory
     *
     *  \param addr Pointer of the memory to be mapped to the accelerator
     *  \param size Size (in bytes) of the host memory to be mapped
     *  \return Error code
     */
    virtual gmacError_t hostAlloc(hostptr_t &addr, size_t size) = 0;

    /**
     * Insert an object into the orphan list
     * \param obj Object to be inserted
     */
    virtual void makeOrphan(memory::Object &obj) = 0;

    /**
     * Maps the given host memory on the accelerator memory
     * \param dst Reference to a pointer where to store the accelerator
     * address of the mapping
     * \param src Host address to be mapped
     * \param size Size of the mapping
     * \param align Alignment of the memory mapping. This value must be a
     * power of two
     * \return Error code
     */
    virtual gmacError_t map(accptr_t &dst, hostptr_t src, size_t size, unsigned align = 1) = 0;

    virtual gmacError_t add_mapping(accptr_t dst, hostptr_t src, size_t count) = 0;


    /** Release GPU-accessible host memory
     *
     *  \param addr Starting address of the host memory to be released
     *  \return Error code
     */
    virtual gmacError_t hostFree(hostptr_t addr) = 0;


    /** Gets the GPU memory address where the given GPU-accessible host
     *  memory pointer is mapped
     *
     *  \param addr Host memory address
     *  \return Device memory address
     */
    virtual accptr_t hostMapAddr(const hostptr_t addr) = 0;


    /**
     * Unmaps the memory previously mapped by map
     * \param addr Host memory allocation to be unmap
     * \param size Size of the unmapping
     * \return Error code
     */
    virtual gmacError_t unmap(hostptr_t addr, size_t size) = 0;

    /**
     * Copies data from system memory to accelerator memory
     * \param acc Destination accelerator pointer
     * \param host Source host pointer
     * \param size Number of bytes to be copied
     * \return Error code
     */
    virtual gmacError_t copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size) = 0;

    /**
     * Copies data from accelerator memory to system memory
     * \param host Destination host pointer
     * \param acc Source accelerator pointer
     * \param size Number of bytes to be copied
     * \return Error code
     */
    virtual gmacError_t copyToHost(hostptr_t host, const accptr_t acc, size_t size) = 0;

    /** Copies data from accelerator memory to accelerator memory
     * \param dst Destination accelerator memory
     * \param src Source accelerator memory
     * \param size Number of bytes to be copied
     * \return Error code
     */
    virtual gmacError_t copyAccelerator(accptr_t dst, const accptr_t src, size_t size) = 0;

    /**
     * Sets the contents of accelerator memory
     * \param addr Pointer to the accelerator memory to be set
     * \param c Value used to fill the memory
     * \param size Number of bytes to be set
     * \return Error code
     */
    virtual gmacError_t memset(accptr_t addr, int c, size_t size) = 0;

    /**
     * Creates an IOBuffer
     * \param size Minimum size of the buffer
     * \param prot Tells whether the requested buffer is going to be read or
     * written on the host
     * \return A pointer to the created IOBuffer or NULL if there is not enough
     *         memory
     */
    virtual IOBuffer &createIOBuffer(size_t size, GmacProtection prot) = 0;

    /**
     * Destroys an IOBuffer
     * \param buffer Pointer to the buffer to be destroyed
     */
    virtual void destroyIOBuffer(IOBuffer &buffer) = 0;

    /** Copies size bytes from an IOBuffer to accelerator memory
     * \param dst Pointer to accelerator memory
     * \param buffer Reference to the source IOBuffer
     * \param size Number of bytes to be copied
     * \param off Offset within the buffer
     */
    virtual gmacError_t bufferToAccelerator(accptr_t dst, IOBuffer &buffer, size_t size, size_t off = 0) = 0;

    /**
     * Copies size bytes from accelerator memory to a IOBuffer
     * \param buffer Reference to the destination buffer
     * \param dst Pointer to accelerator memory
     * \param size Number of bytes to be copied
     * \param off Offset within the buffer
     */
    virtual gmacError_t acceleratorToBuffer(IOBuffer &buffer, const accptr_t dst, size_t size, size_t off = 0) = 0;

    /** Returns the memory information of the accelerator on which the mode runs
     * \param free A reference to a variable to store the memory available on the
     * accelerator
     * \param total A reference to a variable to store the total amount of memory
     * on the accelerator
     */
    virtual void getMemInfo(size_t &free, size_t &total) = 0;

    virtual bool hasIntegratedMemory() const = 0;
    virtual bool hasUnifiedAddressing() const = 0;

#ifdef USE_OPENCL
    virtual gmacError_t acquire(hostptr_t addr) = 0;
    virtual gmacError_t release(hostptr_t addr) = 0;
#endif

    virtual memory::ObjectMap &getAddressSpace() = 0;
    virtual const memory::ObjectMap &getAddressSpace() const = 0;
};

}}

#include "Mode-impl.h"

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
