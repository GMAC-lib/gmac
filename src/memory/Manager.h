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

#ifndef GMAC_MEMORY_MANAGER_H_
#define GMAC_MEMORY_MANAGER_H_

#include "config/common.h"
#include "include/gmac/types.h"
#include "util/Singleton.h"

namespace __impl {

namespace core {
class Process;
class Mode;
class IOBuffer;
}

namespace memory {

class Object;
class Protocol;

typedef std::pair<hostptr_t, GmacProtection> ObjectInfo;
typedef std::list<ObjectInfo> ListAddr;
extern ListAddr AllAddresses;

//! Memory Manager Interface

//! Memory Managers orchestate the data transfers between host and accelerator memories
class GMAC_LOCAL Manager : public __impl::util::Singleton<gmac::memory::Manager> {
    DBC_FORCE_TEST(Manager)
protected:
    /** Process where the memory manager is being used */
    core::Process &proc_;

    /**
     * Allocates a host mapped memory
     * \param mode Execution mode requesting the allocation
     * \param addr Pointer to the variable that will store the begining of the
     * allocated memory
     * \param size Size (in bytes) of the memory to be allocated
     * \return Error code
     */
    gmacError_t hostMappedAlloc(core::Mode &mode, hostptr_t *addr, size_t size);

    /**
     * Default destructor
     */
    virtual ~Manager();
public:
    /**
     * Default constructor
     */
    Manager(core::Process &proc);

    /**
     * Map the given host memory pointer to the accelerator memory. If the given
     * pointer is NULL, host memory is alllocated too.
     * \param mode Execution mode where to allocate memory
     * \param addr Memory address to be mapped or NULL if host memory is requested
     * too
     * \param size Size (in bytes) of shared memory to be mapped 
     * \param flags 
     * \return Error code
     */
    gmacError_t map(core::Mode &mode, hostptr_t *addr, size_t size, int flags);

    gmacError_t remap(core::Mode &mode, hostptr_t old_addr, hostptr_t *new_addr, size_t new_size, int flags);

    /**
     * Unmap the given host memory pointer from the accelerator memory
     * \param mode Execution mode where to allocate memory
     * \param addr Memory address to be unmapped
     * \param size Size (in bytes) of shared memory to be unmapped
     * \return Error code
     */
    gmacError_t unmap(core::Mode &mode, hostptr_t addr, size_t size);

    /**
     * Returns the size of the allocation represented by the given pointer
     * \param mode Execution mode where the memory was allocated
     * \param addr Memory address of the allocation
     * \param size A reference to a variable to store the size of the allocation
     * \return Error code
     */
    gmacError_t getAllocSize(core::Mode &mode, hostptr_t addr, size_t &size) const;

    /**
     * Allocate private shared memory.
     * Memory allocated with this call is only accessible by the accelerator
     * associated to the execution thread requesting the allocation
     * \param mode Execution mode where to allocate memory
     * \param addr Memory address of a pointer to store the host address of the
     * allocated memory
     * \param size Size (in bytes) of shared memory to be allocated
     * \return Error code
     */
    TESTABLE gmacError_t alloc(core::Mode &mode, hostptr_t *addr, size_t size);

    /**
     * Allocate public shared read-only memory.
     * Memory allocated with this call is accessible (read-only) from any
     * accelerator
     * \param mode Execution mode where to allocate memory
     * \param addr Memory address of a pointer to store the host address of the
     * allocated memory
     * \param size Size (in bytes) of shared memory to be allocated
     * \param hint Type of memory (distributed or hostmapped) to be allocated
     * \return Error code
     */
    TESTABLE gmacError_t globalAlloc(core::Mode &mode, hostptr_t *addr, size_t size, GmacGlobalMallocType hint);

    /**
     * Release shared memory
     * \param mode Execution mode where to release the memory
     * \param addr Memory address of the shared memory chunk to be released
     * \return Error code
     */
    TESTABLE gmacError_t free(core::Mode &mode, hostptr_t addr);

    /** Get the accelerator address associated to a shared memory address
     * \param mode Execution mode requesting the translation
     * \param addr Host shared memory address
     * \return Accelerator memory address
     */
    TESTABLE accptr_t translate(core::Mode &mode, hostptr_t addr);

    /**
     * Get the CPU ownership of all objects bound to the current execution mode
     * \param mode Execution mode acquiring the memory objects
     * \param addrs List of addresses to be acquired, or the AllAddresses list if all
     * objects are acquired
     * \return Error code
     */
    gmacError_t acquireObjects(core::Mode &mode, const ListAddr &addrs = AllAddresses);

    /**
     * Release the CPU ownership of all objects bound to the current execution
     * mode
     * \param mode Execution mode releasing the objects
     * \param addrs List of addresses to be released, or the AllAddresses list if all
     * objects are released
     * \return Error code
     */
    gmacError_t releaseObjects(core::Mode &mode, const ListAddr &addrs = AllAddresses);

    /**
     * Notify a memory fault caused by a load operation
     * \param mode Execution mode causing the fault
     * \param addr Host memory address causing the memory fault
     * \return True if the Manager was able to fix the fault condition
     */
    TESTABLE bool signalRead(core::Mode &mode, hostptr_t addr);

    /**
     * Notify a memory fault caused by a store operation
     * \param mode Execution mode causing the fault
     * \param addr Host memory address causing the memory fault
     * \return True if the Manager was able to fix the fault condition
     */
    TESTABLE bool signalWrite(core::Mode &mode, hostptr_t addr);

    /**
     * Copy data from a memory object to an I/O buffer
     * \param mode Execution mode requesting the transfer
     * \param buffer Destionation I/O buffer to copy the data to
     * \param bufferOff Offset within the buffer to copy data to
     * \param addr Host memory address corresponding to a memory object to copy
     * data from
     * \param size Size (in bytes) of the data to be copied
     * \return Error code
     */
    TESTABLE gmacError_t toIOBuffer(core::Mode &mode, core::IOBuffer &buffer, size_t bufferOff, const hostptr_t addr, size_t size);

    /**
     * Copy data from an I/O buffer to a memory object
     * \param mode Execution mode requesting the transfer
     * \param addr Host memory address corresponding to a memory object to copy
     * data to
     * \param buffer Source I/O buffer to copy the data from
     * \param bufferOff Offset within the buffer to copy data from
     * \param size Size (in bytes) of the data to be copied
     * \return Error code
     */
    TESTABLE gmacError_t fromIOBuffer(core::Mode &mode, hostptr_t addr, core::IOBuffer &buffer, size_t bufferOff, size_t size);

    /**
     * Initialize to a given value the contents of a host address of a memory
     * object
     * \param mode Execution mode requesting the operation
     * \param dst Host memory address corresponding to a memory object to set
     * the memory contents
     * \param c Value used to initialize memory
     * \param size Size (in bytes) of the memory to initialize
     * \return Error code
     */
    TESTABLE gmacError_t memset(core::Mode &mode, hostptr_t dst, int c, size_t size);

    /**
     * Copy data from and/or to host memory addresses of memory objects
     * \param mode Execution mode requesting the operation
     * \param dst Destination host memory addrees of a memory objec to copy the
     * data to
     * \param src Source host memory address that might or might not correspond
     * to a memory object
     * \param size Size (in bytes) of the amoun of data to be copied
     */
    TESTABLE gmacError_t memcpy(core::Mode &mode, hostptr_t dst, const hostptr_t src, size_t size);

    gmacError_t flushDirty(core::Mode &mode);
};

}}

#include "Manager-impl.h"

#ifdef USE_DBC
#include "memory/dbc/Manager.h"
#endif

#endif
