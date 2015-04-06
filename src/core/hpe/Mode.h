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

#ifndef GMAC_CORE_HPE_MODE_H_
#define GMAC_CORE_HPE_MODE_H_

#include "config/common.h"

#include "core/Mode.h"
#include "core/hpe/AddressSpace.h"

#ifdef USE_VM
#include "memory/vm/Bitmap.h"
#endif

#include "util/Lock.h"
#include "util/NonCopyable.h"
#include "util/Reference.h"
#include "util/Private.h"

namespace __impl {

namespace memory { class Object; class Block; }

namespace core {

class IOBuffer;

namespace hpe {

class Accelerator;
class Context;
class Kernel;
class KernelLaunch;
class Process;

class GMAC_LOCAL ContextMap : protected std::map<THREAD_T, Context *>, gmac::util::RWLock {
protected:
    typedef std::map<THREAD_T, Context *> Parent;
    Mode &owner_;
public:
    ContextMap(Mode &owner);
    void add(THREAD_T id, Context *ctx);
    Context *find(THREAD_T id);
    void remove(THREAD_T id);
    void clean();
};

/**
 * A Mode represents the address space of a thread in an accelerator. Each
 * thread has one mode per accelerator type in the system
 */
class GMAC_LOCAL Mode : public virtual core::Mode {
    DBC_FORCE_TEST(Mode)
    friend class ContextMap;
    friend class Accelerator;
protected:
    Process &proc_;
    // Must be a pointer since the Mode can change the accelerator on which it is running
    Accelerator *acc_;

    AddressSpace *aSpace_;

    stream_t streamToHost_;
    stream_t streamToAccelerator_;
    stream_t streamLaunch_;
#ifdef USE_VM
    __impl::memory::vm::Bitmap bitmap_;
#endif

    ContextMap contextMap_;

    typedef std::map<gmac_kernel_id_t, Kernel *> KernelMap;
    KernelMap kernels_;

    virtual void switchIn() = 0;
    virtual void switchOut() = 0;

    virtual void reload() = 0;
    virtual Context &getContext() = 0;
    virtual void destroyContext(Context &context) const = 0;

    /**
     * Releases the resources used by the contexts associated to the mode
    */
    TESTABLE void cleanUpContexts();

    /**
     * Releases the resources used by the mode
     *
     * \return gmacSuccess on success. An error code on failure
    */
    TESTABLE gmacError_t cleanUp();

    /**
     * Mode constructor
     *
     * \param proc Reference to the process which the mode belongs to
     * \param acc Reference to the accelerator in which the mode will perform
     *            the allocations
    */
    Mode(Process &proc, Accelerator &acc, AddressSpace &aSpace);

    /**
     * Mode destructor
     */
    virtual ~Mode();

public:
    /**
     * Insert an object into the orphan list
     * \param obj Object to be inserted
     */
    void makeOrphan(memory::Object &obj);

    /**
     * Gets a reference to the accelerator which the mode belongs to
     * \return A reference to the accelerator which the mode belongs to
     */
    Accelerator &getAccelerator() const;

    /**
     * Maps the given host memory on the accelerator memory
     * \param dst Reference to a pointer where to store the accelerator
     * address of the mapping
     * \param src Host address to be mapped
     * \param count Size of the mapping
     * \param align Alignment of the memory mapping. This value must be a
     * power of two
     * \return Error code
     */
    TESTABLE gmacError_t map(accptr_t &dst, hostptr_t src, size_t count, unsigned align = 1);

    gmacError_t
    add_mapping(accptr_t dst, hostptr_t src, size_t count);

    /**
     * Unmaps the memory previously mapped by map
     * \param addr Host memory allocation to be unmap
     * \param count Size of the unmapping
     * \return Error code
     */
    TESTABLE gmacError_t unmap(hostptr_t addr, size_t count);

    /**
     * Copies data from system memory to accelerator memory
     * \param acc Destination accelerator pointer
     * \param host Source host pointer
     * \param count Number of bytes to be copied
     * \return Error code
     */
    TESTABLE gmacError_t copyToAccelerator(accptr_t acc, const hostptr_t host, size_t count);

    /**
     * Copies data from accelerator memory to system memory
     * \param host Destination host pointer
     * \param acc Source accelerator pointer
     * \param count Number of bytes to be copied
     * \return Error code
     */
    TESTABLE gmacError_t copyToHost(hostptr_t host, const accptr_t acc, size_t count);

    /** Copies data from accelerator memory to accelerator memory
     * \param dst Destination accelerator memory
     * \param src Source accelerator memory
     * \param count Number of bytes to be copied
     * \return Error code
     */
    TESTABLE gmacError_t copyAccelerator(accptr_t dst, const accptr_t src, size_t count);

    /**
     * Sets the contents of accelerator memory
     * \param addr Pointer to the accelerator memory to be set
     * \param c Value used to fill the memory
     * \param count Number of bytes to be set
     * \return Error code
     */
    TESTABLE gmacError_t memset(accptr_t addr, int c, size_t count);

    /**
     * Creates a KernelLaunch object that can be executed by the mode
     * \param kernel Handler of the kernel to be launched
     * \param launch Refernce to store the executable kernel
     * \return Reference to the KernelLaunch object
     */
    virtual gmacError_t launch(gmac_kernel_id_t kernel, KernelLaunch *&launch) = 0;

    /**
     * Executes a kernel using a KernelLaunch object
     * \param launch Reference to a KernelLaunch object
     * \return Error code
     */
    virtual gmacError_t execute(KernelLaunch &launch) = 0;

    /**
     * Waits for kernel execution
     * \param launch Reference to KernelLaunch object
     * \return Error code
     */
    gmacError_t wait(KernelLaunch &launch);

    /**
     * Waits for all kernels to finish execution
     * \return Error code
     */
    gmacError_t wait();

    /**
     * Destroys an IOBuffer
     * \param buffer Pointer to the buffer to be destroyed
     */
    virtual void destroyIOBuffer(IOBuffer &buffer) = 0;

    /** Copies count bytes from an IOBuffer to accelerator memory
     * \param dst Pointer to accelerator memory
     * \param buffer Reference to the source IOBuffer
     * \param count Number of bytes to be copied
     * \param off Offset within the buffer
     */
    TESTABLE gmacError_t bufferToAccelerator(accptr_t dst, IOBuffer &buffer, size_t count, size_t off = 0);

    /**
     * Copies count bytes from accelerator memory to a IOBuffer
     * \param buffer Reference to the destination buffer
     * \param dst Pointer to accelerator memory
     * \param count Number of bytes to be copied
     * \param off Offset within the buffer
     */
    TESTABLE gmacError_t acceleratorToBuffer(IOBuffer &buffer, const accptr_t dst, size_t count, size_t off = 0);

    /**
     * Registers a new kernel that can be executed by the owner thread of the mode
     *
     * \param k A key that identifies the kernel object
     * \param kernel A reference to the kernel to be registered
     */
    TESTABLE void registerKernel(gmac_kernel_id_t k, Kernel &kernel);

    /**
     * Returns the kernel name identified by k
     *
     * \param k A key that identifies the kernel object
     */
    TESTABLE std::string getKernelName(gmac_kernel_id_t k) const;

    /**
     * Moves the mode to accelerator acc
     * \param acc Accelerator to move the mode to
     * \return Error code
     */
    TESTABLE gmacError_t moveTo(Accelerator &acc);

    /**
     * Returns the process which the mode belongs to
     * \return A reference to the process which the mode belongs to
     */
    Process &getProcess();

    /** Returns the process which the mode belongs to
     * \return A constant reference to the process which the mode belongs to
     */
    const Process &getProcess() const;

    /** Returns the memory information of the accelerator on which the mode runs
     * \param free A reference to a variable to store the memory available on the
     * accelerator
     * \param total A reference to a variable to store the total amount of memory
     * on the accelerator
     */
    void getMemInfo(size_t &free, size_t &total);

#ifdef USE_VM
    memory::vm::Bitmap &getDirtyBitmap();
    const memory::vm::Bitmap &getDirtyBitmap() const;
#endif

    /**
     * Waits for pending transfers before performing a kernel call
     *
     * \return gmacSuccess on success, an error code otherwise
     */
    gmacError_t prepareForCall();

    stream_t eventStream();

    /**
     * Tells if the accelerator on which the Mode is running shares memory with the CPU
     *
     * \return A boolean that tells if the accelerator on which the Mode is running shares memory with the CPU
     */
    bool hasIntegratedMemory() const;
    bool hasUnifiedAddressing() const;

    memory::ObjectMap &getAddressSpace();
    const memory::ObjectMap &getAddressSpace() const;

    void setAddressSpace(AddressSpace &aSpace);
};

}}}

#include "Mode-impl.h"

#ifdef USE_DBC
#include "core/hpe/dbc/Mode.h"
#endif

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
