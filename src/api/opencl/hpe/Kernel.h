/* Copyright (c) 2009, 2010 University of Illinois
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


#ifndef GMAC_API_OPENCL_HPE_KERNEL_H_
#define GMAC_API_OPENCL_HPE_KERNEL_H_

#if defined(__APPLE__)
#   include <OpenCL/cl.h>
#else
#   include <CL/cl.h>
#endif

#include <list>

#include "config/common.h"

#include "api/opencl/Tracer.h"
#include "core/hpe/Kernel.h"
#include "util/NonCopyable.h"

namespace __impl { namespace opencl { namespace hpe {

class Mode;

class KernelConfig;
class KernelLaunch;

/** A kernel that can be executed by an OpenCL accelerator */
class GMAC_LOCAL Kernel : public gmac::core::hpe::Kernel {
    friend class KernelLaunch;
protected:
    /** OpenCL kernel object */
    cl_kernel f_;
    /** Number of arguments requried by the kernel */
    unsigned nArgs_;
public:
    /**
     * Default constructor
     * \param k Kernel descriptor that describes the kernel
     * \param kernel OpenCL kernel containing the kernel code
     */
    Kernel(const core::hpe::KernelDescriptor & k, cl_kernel kernel);

    /**
     * Default destructor
     */
    ~Kernel();

    /**
     * Get a kernel that can be executed by an execution mode
     * \param mode Execution mode capable of executing the kernel
     * \param stream OpenCL queue where the kernel can be executed
     * \return Invocable kernel
     */
    KernelLaunch *launch(Mode &mode, cl_command_queue stream);

    unsigned getNArgs() const;
};

static const size_t MAX_DIMS = 5;

/** An OpenCL kernel that can be executed */
class GMAC_LOCAL KernelLaunch :
    public core::hpe::KernelLaunch,
    public util::NonCopyable {
    friend class Kernel;

    typedef std::pair<cl_mem, unsigned> CLMemRef;
    typedef std::map<hostptr_t, CLMemRef> MapSubBuffer;
    class MapGlobalSubBuffer :
        protected std::map<__impl::core::hpe::Mode *, MapSubBuffer>,
        protected gmac::util::SpinLock
    {
    public:
        typedef std::map<__impl::core::hpe::Mode *, MapSubBuffer> Parent;
        typedef Parent::iterator iterator;
        MapGlobalSubBuffer() :
            gmac::util::SpinLock("SubBuffer Lock")
        {
        }

        iterator findMode(__impl::core::hpe::Mode &mode)
        {
            lock();
            Parent::iterator itGlobalMap = this->find(&mode);
            if (itGlobalMap == this->end()) {
                this->insert(Parent::value_type(&mode, MapSubBuffer()));
                itGlobalMap = this->find(&mode);
            }
            unlock();
            return itGlobalMap;
        }
    };

    typedef std::pair<__impl::core::hpe::Mode *, MapSubBuffer::iterator> CacheEntry;
    typedef std::map<hostptr_t, CacheEntry> CacheSubBuffer;

protected:
    /** OpenCL kernel code */
    cl_kernel f_;
    /** OpenCL command queue where the kernel is executed */
    cl_command_queue stream_;

    cl_event event_;

    /** Number of dimensions the kernel will execute */
    cl_uint workGlobalDim_;
    cl_uint workLocalDim_;
    cl_uint offsetDim_;
    /** Index offsets for each kernel dimension */
    size_t globalWorkOffset_[MAX_DIMS];
    /** Number of elements per kernel dimension */
    size_t globalWorkSize_[MAX_DIMS];
    /** Number of elements per kernel work-group dimension */
    size_t localWorkSize_[MAX_DIMS];

    /** Tracer */
    KernelExecution trace_;

    /** Subbuffers created to allow pointer arithmetic */
    static MapGlobalSubBuffer mapSubBuffer_;

    /** Cache of subbuffers created to allow pointer arithmetic */
    CacheSubBuffer cacheSubBuffer_;

    /**
     * Default constructor
     * \param mode Execution mode executing the kernel
     * \param k OpenCL kernel object
     * \param stream OpenCL command queue executing the kernel
     */
    KernelLaunch(Mode &mode, const Kernel & k, cl_command_queue stream);
public:
    /**
     * Default destructor
     */
    ~KernelLaunch();

    /**
     * Execute the kernel
     * \return Error code
     */
    gmacError_t execute();

    /**
     * Get the OpenCL event that defines when the kernel execution is complete
     * \return OpenCL event which is completed after the kernel execution is done
     */
    cl_event getCLEvent();

    /**
     * Set the configuration parameters for a kernel
     * \param workDim Number of dimensions for the kernel
     * \param globalWorkOffset Index offsets for each dimenssion
     * \param globalWorkSize Number of elements per dimension
     * \param localWorkSize Number of work-group items per dimension
     */
    void setConfiguration(cl_uint workDim, const size_t *globalWorkOffset,
        const size_t *globalWorkSize, const size_t *localWorkSize);

    /**
     * Set a new argument for the kernel
     * \param arg Pointer to the value for the argument
     * \param size Size (in bytes) of the argument
     * \param index Index of the argument in the argument list
     */
    gmacError_t setArgument(const void * arg, size_t size, unsigned index);

#if 0
    /**
     * Tells if the kernel launch object already has a subbuffer for the given pointer
     * \param ptr Pointer to be used as an argument in a kernel call
     * \return A boolean that tells if the kernel launch object already has a subbuffer for the given pointer
     */
    bool hasSubBuffer(hostptr_t ptr) const;
#endif

    /**
     * Returns the subbuffer associated to the given pointer
     * \param ptr Pointer to be used as an argument in a kernel call
     * \return The subbuffer associated to the given pointer
     */
    cl_mem getSubBuffer(__impl::opencl::hpe::Mode &mode, hostptr_t ptr, accptr_t accPtr, size_t size);

#if 0
    /**
     * Sets the subbuffer associated to the given pointer
     * \param ptr Pointer to be used as an argument in a kernel call
     * \param subMeme The subbuffer associated to the given pointer
     */
    void setSubBuffer(hostptr_t ptr, cl_mem subMem);
#endif
};

}}}

#include "Kernel-impl.h"

#endif
