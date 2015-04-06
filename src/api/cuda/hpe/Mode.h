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

#ifndef GMAC_API_CUDA_HPE_MODE_H_
#define GMAC_API_CUDA_HPE_MODE_H_

#include <cuda.h>
#include <vector_types.h>

#include "config/common.h"
#include "config/config.h"

#include "core/hpe/Mode.h"
#include "api/cuda/Mode.h"
#include "api/cuda/hpe/ContextFactory.h"

#include "util/GMACBase.h"

#include "Module.h"

namespace __impl {

namespace core {
class IOBuffer;
namespace hpe {
    class AddressSpace;
}
}

namespace util { namespace allocator {
    class Buddy;
}}

namespace cuda { namespace hpe {

class Context;

class GMAC_LOCAL ContextLock : public gmac::util::Lock {
    friend class Mode;
public:
    ContextLock() : gmac::util::Lock("Context") {}
};

class Texture;
class Accelerator;

//! A Mode represents a virtual CUDA accelerator on an execution thread
class GMAC_LOCAL Mode :
    util::GMACBase<Mode>,
    public ContextFactory,
    public gmac::core::hpe::Mode,
    public virtual cuda::Mode {

    DBC_FORCE_TEST(Mode)

    friend class Switch;
    friend class ModeFactory;
protected:
#ifdef USE_MULTI_CONTEXT
    //! Associated CUDA context
    CUcontext cudaCtx_;
#endif
    util::allocator::Buddy *ioMemoryRead_;
    util::allocator::Buddy *ioMemoryWrite_;

    //! Switch to accelerator mode
    void switchIn();

    //! Switch back to CPU mode
    void switchOut();

    //! Get the main mode context
    /*!
        \return Main mode context
    */
    core::hpe::Context &getContext();
    Context &getCUDAContext();
    void destroyContext(core::hpe::Context &context) const;

#ifdef USE_MULTI_CONTEXT
    //! CUDA modules active on this mode
    ModuleVector modules_;
#else
    //! CUDA modules active on this mode
    ModuleVector *modules_;
#endif

    //! Load CUDA modules and kernels
    void load();

    //! Reload CUDA kernels
    void reload();

    //! Default constructor
    /*!
        \param proc Process where the mode is attached
        \param acc Virtual CUDA accelerator where the mode is executed
        \param aSpace Address space in which the mode will run
    */
    Mode(core::hpe::Process &proc, Accelerator &acc, core::hpe::AddressSpace &aSpace);

    //! Default destructor
    virtual ~Mode();

public:
    /*!
        \param addr Memory address of the pointer where the starting host memory address will be stored
        \param size Size (in bytes) of the host memory to be allocated
        \return Error code
    */
    gmacError_t hostAlloc(hostptr_t &addr, size_t size);
    //! Release GPU-accessible host memory
    /*!
        \param addr Starting address of the host memory to be released
        \return Error code
    */
    gmacError_t hostFree(hostptr_t addr);

    //! Get the GPU memory address where GPU-accessible host memory is mapped
    /*!
        \param addr Host memory address
        \return Device memory address
    */
    accptr_t hostMapAddr(const hostptr_t addr);

    gmacError_t launch(gmac_kernel_id_t id, core::hpe::KernelLaunch *&kernel);

    //! Execute a kernel on the accelerator
    /*!
        \param launch Structure defining the kernel to be executed
        \return Error code
    */
    gmacError_t execute(core::hpe::KernelLaunch &launch);

    core::IOBuffer &createIOBuffer(size_t size, GmacProtection prot);
    void destroyIOBuffer(core::IOBuffer &buffer);

    gmacError_t call(dim3 Dg, dim3 Db, size_t shared, cudaStream_t tokens);
    gmacError_t argument(const void *arg, size_t size, off_t offset);

    const Variable *constant(gmacVariable_t key) const;
    const Variable *variable(gmacVariable_t key) const;
    const Variable *constantByName(std::string name) const;
    const Variable *variableByName(std::string name) const;
    const Texture *texture(gmacTexture_t key) const;

    Accelerator &getAccelerator() const;

    gmacError_t waitForEvent(CUevent event, bool fromCUDA);
    gmacError_t eventTime(uint64_t &t, CUevent start, CUevent end);
};

}}}

#include "Mode-impl.h"

#ifdef USE_DBC
#include "dbc/Mode.h"
#endif

#endif
