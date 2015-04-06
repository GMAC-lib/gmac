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
WITH THE SOFTWARE.
*/

#ifndef GMAC_API_OPENCL_HPE_ACCELERATOR_H_
#define GMAC_API_OPENCL_HPE_ACCELERATOR_H_

#if defined(__APPLE__)
#   include <OpenCL/cl.h>
#else
#   include <CL/cl.h>
#endif

#include <list>
#include <map>
#include <utility>
#include <vector>

#include "config/common.h"

#include "api/opencl/Tracer.h"
#include "api/opencl/hpe/ModeFactory.h"
#include "core/hpe/Accelerator.h"
#include "util/Lock.h"
#include "util/UniquePtr.h"

namespace __impl { namespace opencl {

class IOBuffer;

namespace hpe {

class Mode;

class KernelLaunch;
class Kernel;

/** A list of command queues */
class GMAC_LOCAL CommandList :
    protected std::list<cl_command_queue>,
    protected gmac::util::RWLock {
protected:
    /** Base type from STL */
    typedef std::list<cl_command_queue> Parent;
public:
    /** Default constructor */
    CommandList() : RWLock("CommandList") {}
    /** Default destructor */
    virtual ~CommandList();

    /** Add a command queue to the list
     * \param stream Command queue to be inserted
     */
    void add(cl_command_queue stream);

    /** Remove a command queue from the list
     * \param stream Command queue to be removed
     */
    void remove(cl_command_queue stream);

    /** Get the command queue from the front of the list
     * \return Command queue at the fron of the list
     */
    cl_command_queue &front();

    /** Wait for all command queue in the list to finish execution
     * \return Error code
     */
    cl_int sync() const;

    bool empty() const;
};

/** A map of host memory addresses associated to OpenCL memory objects */
class GMAC_LOCAL HostMap :
    protected std::map<hostptr_t, std::pair<cl_mem, size_t> >,
    protected gmac::util::RWLock {
protected:
    /** Base type from STL */
    typedef std::map<hostptr_t, std::pair<cl_mem, size_t> > Parent;
public:
    /** Default constructor */
    HostMap() : RWLock("HostMap") { }
    /** Default destructor */
    virtual ~HostMap();

    /** Insert a new entry in the map
     * \param host Host memory address
     * \param acc OpenCL object
     * \param size (in bytes) of the object
     */
    void insert(hostptr_t host, cl_mem acc, size_t size);

    /** Remove an entry from the map
     * \param host Host memory address of the entry
     */
    void remove(hostptr_t host);

    /** Get the OpenCL memory object associated to a host memory address
     * \param host Host memory address
     * \param acc Reference to store the associated OpenCL memory object
     * \param size Reference to the size (in bytes) of the OpenCL memory object
     * \return True if the translation succeeded
     */
    bool translate(const hostptr_t host, cl_mem &acc, size_t &size) const;
};

/** A pool of OpenCL buffers */
class GMAC_LOCAL CLBufferPool :
    protected std::map<size_t, std::list<std::pair<cl_mem, hostptr_t> > >,
    protected gmac::util::Lock {

    typedef std::list<std::pair<cl_mem, hostptr_t> > CLMemList;
    typedef std::map<size_t, CLMemList> CLMemMap;
public:
    /** Constructs a pool of OpenCL buffers */
    CLBufferPool();

    /**
     * Gets an OpenCL buffer from the pool
     *
     * \param size Requested size (in bytes) for the buffer
     * \param mem Reference to store the cl_mem descriptor of the buffer
     * \param addr Reference to store the host address of the buffer
     *
     * \return true if a buffer was found, false otherwise
     */
    bool getCLMem(size_t size, cl_mem &mem, hostptr_t &addr);

    /**
     * Inserts an OpenCL buffer to the pool
     *
     * \param size Size (in bytes) of the buffer
     * \param mem cl_mem descriptor of the buffer
     * \param addr Host address of the buffer
     */
    void putCLMem(size_t size, cl_mem mem, hostptr_t addr);

    /**
     * Releases the OpenCL buffers in the pool
     *
     * \param stream OpenCL stream to enqueue unmaps
     */
    void cleanUp(stream_t stream);
};

/** An OpenCL capable accelerator */
class GMAC_LOCAL Accelerator :
    protected ModeFactory,
    public gmac::util::SpinLock,
    public gmac::core::hpe::Accelerator {

    DBC_FORCE_TEST(Accelerator);
protected:
    typedef std::map<Accelerator *, std::vector<cl_program> > AcceleratorMap;
    /** Map of the OpenCL accelerators in the system and the associated OpenCL programs */
    static AcceleratorMap *Accelerators_;
    /** Host memory allocations associated to any OpenCL accelerator */
    static HostMap *GlobalHostAlloc_;

    CLBufferPool clMemRead_;
    CLBufferPool clMemWrite_;

    /** OpenCL context associated to the accelerator */
    cl_context ctx_;
    /** OpenCL device ID for the accelerator */
    cl_device_id device_;

    /** OpenCL major version supported by the accelerator */
    unsigned major_;
    /** OpenCL minor version supported by the accelerator */
    unsigned minor_;

    /** List of command queues associated to the accelerator */
    CommandList cmd_;
    /** Host memory allocations associated to the accelerator */
    HostMap localHostAlloc_;

    /** Tracer for data communications */
    DataCommunication trace_;

    size_t allocatedMemory_;

    /** Is Accelerator information initialized */
    bool isInfoInitialized_;
    /** String containing the accelerator name */
    char *acceleratorName_;
    /** String containing the vendor name */
    char *vendorName_;
    /** Max workgroup sizes for the accelerator */
    size_t *maxSizes_;

public:
    /** Default constructor
     * \param n Accelerator number
     * \param context OpenCL context the accelerator belongs to
     * \param device OpenCL device ID for the accelerator
     * \param major OpenCL major version supported by the accelerator
     * \param minor OpenCL minor version supported by the accelerator
     */
    Accelerator(int n, cl_context context, cl_device_id device, unsigned major, unsigned minor);
    /** Default destructor */
    virtual ~Accelerator();

    /**
     * Get the OpenCL device ID associated to the accelerator
     * \return OpenCL device ID
     */
    cl_device_id device() const;

    /**
     * Initialize the accelerator global data structures
     */
    static void init();

    /**
     * Get a GMAC error associated to an OpenCL error code
     * \param r OpenCL error code
     * \return GMAC error code
     */
    static gmacError_t error(cl_int r);

    /**
     * Add a new accelerator
     * \param acc Accelerator to be added
     */
    static void addAccelerator(Accelerator &acc);

    /**
     * Check for OpenCL code embedded in the binary
     * \return Error code
     */
    static gmacError_t prepareEmbeddedCLCode();

    /**
     * Make source OpenCL kernel available to all accelerators
     * \param code OpenCL source code
     * \param flags Compilation flags
     * \return Error code
     */
    static gmacError_t prepareCLCode(const char *code, const char *flags);

    /**
     * Make binary OpenCL kernel available to all accelerators
     * \param binary OpenCL binary code
     * \param size Size (in bytes) of the OpenCL binary code
     * \param flags Compilation flags
     * \return Error Code
     */
    static gmacError_t prepareCLBinary(const unsigned char *binary, size_t size, const char *flags);

    /**
     * Get a kernel from its ID
     * \param k Kernel ID
     * \return Kernel object
     */
    Kernel *getKernel(gmac_kernel_id_t k);

    /**
     *  Create a new execution mode for this accelerator
     * \param proc Process where to bind the execution mode
     * \param aSpace Address space to be used by the mode
     * \return Execution mode
     */
    core::hpe::Mode *createMode(core::hpe::Process &proc, core::hpe::AddressSpace &aSpace);

    /**
     *  Get the OpenCL context associated to the accelerator
     * \return OpenCL context
     */
    const cl_context getCLContext() const;

    /**
     *  Get the OpenCL context associated to the accelerator
     * \return OpenCL context
     */
    cl_context getCLContext();

    /**
     * Allocate pinned accelerator-accessible host memory
     * \param addr Reference to store the memory address of the allocated memory
     * \param size Size (in bytes) of the memory to be allocated
     * \return Error code
     */
    gmacError_t hostAlloc(hostptr_t &addr, size_t size);

    /**
     * Allocate a host-accessible OpenCL buffer
     * \param mem Reference to store the cl_mem descriptor of the buffer
     * \param addr Reference to store the host address of the OpenCL buffer
     * \param size Size (in bytes) of the memory to be allocated
     * \param prot Tells wether the CL Buffer is going to be read or written
     * from the host
     * \return Error code
     */
    gmacError_t allocCLBuffer(cl_mem &mem, hostptr_t &addr, size_t size, GmacProtection prot);

    /**
     * Release pinned accelerator-accessible host memory
     * \param addr Host memory address to be released
     * \return Error code
     */
    gmacError_t hostFree(hostptr_t addr);

    /**
     * Free a host-accessible OpenCL buffer
     * \param mem cl_mem descriptor of the buffer
     * \param addr Host address of the OpenCL buffer
     * \param size Size (in bytes) of the memory to be allocated
     * \param prot Tells wether the CL Buffer is going to be read or written
     * \return Error code
     */
    gmacError_t freeCLBuffer(cl_mem mem, hostptr_t addr, size_t size, GmacProtection prot);

    /**
     * Get the accelerator memory address where pinned host memory can be accessed
     * \param addr Host memory address to be mapped to the accelerator
     * \return Accelerator memory address
     */
    accptr_t hostMapAddr(hostptr_t addr);

    /**
     * Executes a kernel in the accelerator
     * \param stream OpenCL command queue
     * \param kernel OpenCL kernel to execute
     * \param workDim Number of dimensions in the work group
     * \param offset Offset for the kernel
     * \param globalSize Global size of the kernel to execute
     * \param localSize Local size of the kernel to execute
     * \param event OpenCL event to notify the end of execution
     * \return Error code
     */
    gmacError_t execute(cl_command_queue stream, cl_kernel kernel, cl_uint workDim,
        const size_t *offset, const size_t *globalSize, const size_t *localSize, cl_event *event);

    /**
     * Gets the default OpenCL command queue
     * \return OpenCL command queue
     */
    cl_command_queue getCLstream();

    /**
     * Create an OpenCL command queue
     * \return OpenCL command queue
     */
    cl_command_queue createCLstream();

    /**
     * Destroy an OpenCL command queue
     * \param stream OpenCL command queue to be destroyed
     */
    void destroyCLstream(cl_command_queue stream);

    /**
     * Query for the state of an OpenCL command queue
     * \param stream OpenCL command queue to query for its state
     * \return OpenCL command queue state
     */
    cl_int queryCLstream(cl_command_queue stream);

    /**
     * Wait for all commands in a command queue to be completed
     * \param stream OpenCL command queue
     * \return Error code
     */
    gmacError_t syncStream(stream_t stream);

    /**
     * Query for the state of an OpenCL event
     * \param event OpenCL event
     * \return OpenCL event status
     */
    cl_int queryCLevent(cl_event event);

    /**
     * Wait for an OpenCL event to be completed
     * \param event OpenCL event
     * \return Error code
     */
    gmacError_t syncCLevent(cl_event event);

    /**
     * Calculate the time elapsed between to events happend
     * \param t Reference to store the elapsed time
     * \param start Event when the elapsed time start
     * \param end Event when the elapsed time end
     * \return Error code
     */
    gmacError_t timeCLevents(uint64_t &t, cl_event start, cl_event end);

    /**
     * Execute a kernel
     * \param launch Descriptor of the kernel to be executed
     * \return Error code
     */
    gmacError_t execute(KernelLaunch &launch);

    /**
     * Return OpenCL major number
     * \return OpenCL major number
     */
    unsigned getMajor() const;

    /**
     * Return OpenCL minor number
     * \return OpenCL minor number
     */
    unsigned getMinor() const;


    /* core/hpe/Accelerator.h Interface */
    gmacError_t map(accptr_t &dst, hostptr_t src, size_t size, unsigned align = 1);
    gmacError_t add_mapping(accptr_t dst, hostptr_t src, size_t size);

    TESTABLE gmacError_t unmap(hostptr_t addr, size_t size);

    gmacError_t sync();

    TESTABLE gmacError_t copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size, core::hpe::Mode &mode);
    TESTABLE gmacError_t copyToHost(hostptr_t host, const accptr_t acc, size_t size, core::hpe::Mode &mode);

    TESTABLE gmacError_t copyAccelerator(accptr_t dst, const accptr_t src, size_t size, stream_t stream);
    gmacError_t memset(accptr_t addr, int c, size_t size, stream_t stream);
    void getMemInfo(size_t &free, size_t &total) const;
    void getAcceleratorInfo(GmacAcceleratorInfo &info);

    gmacError_t acquire(hostptr_t addr);
    gmacError_t release(hostptr_t addr);
};

}}}

#include "Accelerator-impl.h"

#ifdef USE_DBC
#include "dbc/Accelerator.h"
#endif


#endif
