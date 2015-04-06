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

#ifndef GMAC_API_OPENCL_LITE_MODE_H_
#define GMAC_API_OPENCL_LITE_MODE_H_

#if defined(__APPLE__)
#   include <OpenCL/cl.h>
#else
#   include <CL/cl.h>
#endif

#include <map>
#include <set>

#include "config/common.h"
#include "config/config.h"

#include "api/opencl/Mode.h"
#include "api/opencl/Tracer.h"
#include "core/AllocationMap.h"
#include "memory/ObjectMap.h"
#include "util/Atomics.h"
#include "util/Private.h"

namespace __impl {

namespace core {
class IOBuffer;
}

namespace opencl { namespace lite {

class GMAC_LOCAL QueueSet :
    protected std::set<cl_command_queue>,
    public gmac::util::RWLock {
protected:
    typedef std::set<cl_command_queue> Parent;
public:
    QueueSet();
    virtual ~QueueSet();

    void insert(cl_command_queue queue);
    bool exists(cl_command_queue queue);
    void remove(cl_command_queue queue);
};

class Mode;

//! A Lock that can be used by Mode
class GMAC_LOCAL ModeLock : public gmac::util::Lock {
protected:
	friend class __impl::opencl::lite::Mode;
    inline ModeLock() : gmac::util::Lock("ModeLock") {};
    inline void lock() { return util::Lock::lock(); }
    inline void unlock() { return util::Lock::unlock(); }
};

//! A Mode represents a virtual OpenCL accelerator on an execution thread
class GMAC_LOCAL Mode :
    public opencl::Mode,
    public core::Mode
{
    friend class core::IOBuffer;
    friend class ModeMap;

protected:
    ModeLock queue_;

    cl_context context_;
    typedef std::map<cl_command_queue, cl_device_id> StreamMap;
    StreamMap streams_;
    QueueSet queues_;
    cl_command_queue active_;

    memory::ObjectMap map_;
    gmac::core::AllocationMap allocations_;

    DataCommunication trace_;

    gmacError_t error(cl_int) const;
public:
    /**
     * Default constructor
     *  \param ctx A OpenCL context
     *  \param numDevices Number of OpenCL devices in the execution mode
     *  \param devices Array of OpenCL devices in the execution mode
     */
    Mode(cl_context ctx, cl_uint numDevices, const cl_device_id *devices);

    /**
     * Default destructor
     */
    virtual ~Mode();

    /**
     * Add a new command queue to the active set
     * \param queue New command queue in the mode
     */
    void addQueue(cl_command_queue queue);

    /**
     * Set the active command queue
     * \param queue Command queue to set as active
     * \return Error code
     */
    gmacError_t setActiveQueue(cl_command_queue queue);

    /**
     * Deactivates the command queue
     */
    void deactivateQueue();

    /**
     * Removes a command queue from the mode
     * \param queue Command queue to be removed
     */
    void removeQueue(cl_command_queue queue);

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
    gmacError_t map(accptr_t &dst, hostptr_t src, size_t size, unsigned align = 1);
    gmacError_t add_mapping(accptr_t dst, hostptr_t src, size_t size);

    /**
     * Allocate GPU-accessible host memory
     * \param addr Pointer of the memory to be mapped to the accelerator
     * \param size Size (in bytes) of the host memory to be mapped
     * \return Error code
     */
    gmacError_t hostAlloc(hostptr_t &addr, size_t size);

    /**
     * Release GPU-accessible host memory
     *
     *  \param addr Starting address of the host memory to be released
     *  \return Error code
     */
    gmacError_t hostFree(hostptr_t addr);

    /** Gets the GPU memory address where the given GPU-accessible host
     *  memory pointer is mapped
     *
     *  \param addr Host memory address
     *  \return Device memory address
     */
    accptr_t hostMapAddr(const hostptr_t addr);

    /**
     * Unmaps the memory previously mapped by map
     * \param addr Host memory allocation to be unmap
     * \param size Size of the unmapping
     * \return Error code
     */
    gmacError_t unmap(hostptr_t addr, size_t size);


    /** Create an IO buffer to sent / receive data from the accelerator
     *
     * \param size Size (in bytes) of the IO buffer
     * \param prot Tells whether the requested buffer is going to be read or
     * written on the host
     *
     * \return Pointer to the created I/O buffer or NULL if not enough memory
     */
    core::IOBuffer &createIOBuffer(size_t size, GmacProtection prot);

    /**
     * Destroy (release) an I/O buffer
     *
     * \param buffer I/O buffer to be released
     */
    void destroyIOBuffer(core::IOBuffer &buffer);

    /** Send data from an I/O buffer to the accelerator
     *
     *  \param dst Accelerator memory where data will be written to
     *  \param buffer I/O buffer where data will be read from
     *  \param size Size (in bytes) of the data to be copied
     *  \param off Offset (in bytes) in the I/O buffer where to start reading data from
     *  \return Error code
     */
    gmacError_t bufferToAccelerator(accptr_t dst, core::IOBuffer &buffer, size_t size, size_t off = 0);

    /** Fill I/O buffer with data from the accelerator
     *
     *  \param buffer I/O buffer where data will be stored
     *  \param src Accelerator memory where the data will be read from
     *  \param size Size (in bytes) of the data to be copied
     *  \param off Offset (in bytes) in the I/O buffer where to start writing data to
     *  \return Error code
     */
    gmacError_t acceleratorToBuffer(core::IOBuffer &buffer, const accptr_t src, size_t size, size_t off = 0);

    /** Get the accelerator stream where events are recorded
     *
     *   \return Command queue where events are recorded
     */
    cl_command_queue eventStream();


    /** Block the CPU thread until an event happens
     *
     *  \param event Event to wait for
     *  \return Error code
     */
    gmacError_t waitForEvent(cl_event event);

    /** Get the current (active) execution mode
     *  \return Current (active) execution mode or NULL if no mode is active
     */
    static Mode & getCurrent();

    /**
     * Get the time elapsed between tow events
     * \param t Reference to store the elapsed time
     * \param start Starting event
     * \param end Ending event
     */
    gmacError_t eventTime(uint64_t &t, cl_event start, cl_event end);

    /**
     * Copies data from system memory to accelerator memory
     * \param acc Destination accelerator pointer
     * \param host Source host pointer
     * \param size Number of bytes to be copied
     * \return Error code
     */
    gmacError_t copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size);

    /**
     * Copies data from accelerator memory to system memory
     * \param host Destination host pointer
     * \param acc Source accelerator pointer
     * \param size Number of bytes to be copied
     * \return Error code
     */
    gmacError_t copyToHost(hostptr_t host, const accptr_t acc, size_t size);

    /** Copies data from accelerator memory to accelerator memory
     * \param dst Destination accelerator memory
     * \param src Source accelerator memory
     * \param size Number of bytes to be copied
     * \return Error code
     */
    gmacError_t copyAccelerator(accptr_t dst, const accptr_t src, size_t size);

    /**
     * Sets the contents of accelerator memory
     * \param addr Pointer to the accelerator memory to be set
     * \param c Value used to fill the memory
     * \param size Number of bytes to be set
     * \return Error code
     */
    gmacError_t memset(accptr_t addr, int c, size_t size);

    /**
     * Insert an object into the orphan list
     * \param obj Object to be inserted
     */
    void makeOrphan(memory::Object &obj);

#if 0
    /**
     * Releases the ownership of the objects of the mode to the accelerator
     * and waits for pending transfers
     * \return Error code
     */
    gmacError_t acquireObjects();
#endif

    /** Returns the memory information of the accelerator on which the mode runs
     * \param free A reference to a variable to store the memory available on the
     * accelerator
     * \param total A reference to a variable to store the total amount of memory
     * on the accelerator
     */
    void getMemInfo(size_t &free, size_t &total);

    bool hasIntegratedMemory() const;
    bool hasUnifiedAddressing() const;

    memory::ObjectMap &getAddressSpace();
    const memory::ObjectMap &getAddressSpace() const;

    gmacError_t acquire(hostptr_t addr);
    gmacError_t release(hostptr_t addr);
};

}}}

#if defined(USE_DBC)
namespace __dbc { namespace opencl { namespace lite {
typedef __impl::opencl::lite::Mode Mode;
}}}
#endif

#include "Mode-impl.h"

#endif
