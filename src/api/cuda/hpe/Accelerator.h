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

#ifndef GMAC_API_CUDA_HPE_ACCELERATOR_H_
#define GMAC_API_CUDA_HPE_ACCELERATOR_H_

#include <vector_types.h>

#include <list>
#include <set>

#include "config/common.h"
#include "core/hpe/Mode.h"
#include "core/hpe/Accelerator.h"
#include "api/cuda/hpe/ModeFactory.h"
#include "util/Lock.h"

#include "Module.h"

namespace __impl { namespace cuda {
class IOBuffer;

namespace hpe {
class ModuleDescriptor;

class GMAC_LOCAL Switch {
public:
    static void in(Mode &mode);
    static void out(Mode &mode);
};

typedef CUstream Stream;
class Accelerator;
class GMAC_LOCAL AcceleratorLock : protected gmac::util::SpinLock {
    friend class Accelerator;
public:
    AcceleratorLock(const char *name) : gmac::util::SpinLock(name) {}
};

class GMAC_LOCAL AlignmentMap : public std::map<CUdeviceptr, CUdeviceptr>, public gmac::util::RWLock {
    friend class Accelerator;
public:
    AlignmentMap() : gmac::util::RWLock("Aligment") {}
    ~AlignmentMap() { lockWrite(); }
};

class GMAC_LOCAL Accelerator :
    public ModeFactory,
    public gmac::core::hpe::Accelerator {
    DBC_FORCE_TEST(Accelerator)
    friend class Switch;
protected:
    CUdevice device_;

    int major_;
    int minor_;

    AlignmentMap alignMap_;

#ifdef USE_VM
#ifndef USE_MULTI_CONTEXT
    Mode *lastMode_;
#endif
#endif
    /** Is Accelerator information initialized */
    bool isInfoInitialized_;
    /** String containing the accelerator name */
    static const unsigned MaxAcceleratorNameLength = 256;
    char acceleratorName_[256];
    /** Max workgroup sizes for the accelerator */
    size_t maxSizes_[3];

#ifdef USE_MULTI_CONTEXT
    static util::Private<CUcontext> Ctx_;
#else
    CUcontext ctx_;
    ModuleVector modules_;
#endif

#if defined(USE_TRACE)
    CUevent start_, end_;
#endif

#if not defined(USE_CUDA_SET_CONTEXT)
    AcceleratorLock contextLock_;
#endif

    void pushContext() const;
    void popContext() const;

public:
    Accelerator(int n, CUdevice device);
    virtual ~Accelerator();

#ifndef USE_MULTI_CONTEXT
#ifdef USE_VM
    cuda::hpe::Mode *getLastMode();
    void setLastMode(cuda::hpe::Mode &mode);
#endif
#endif

    CUdevice device() const;

    static void init();

    __impl::core::hpe::Mode *createMode(core::hpe::Process &proc, __impl::core::hpe::AddressSpace &aSpace);

#ifdef USE_MULTI_CONTEXT
    CUcontext createCUcontext();
    void destroyCUcontext(CUcontext ctx);

    void setCUcontext(CUcontext *ctx);
    ModuleVector createModules();
    void destroyModules(ModuleVector & modules);
#else
    const CUcontext getCUcontext() const;

    ModuleVector &createModules();
#endif

    int major() const;
    int minor() const;

    gmacError_t map(accptr_t &dst, hostptr_t src, size_t size, unsigned align = 1);
    gmacError_t add_mapping(accptr_t dst, hostptr_t src, size_t size);

    gmacError_t unmap(hostptr_t addr, size_t size);

#if CUDA_VERSION >= 4000
    gmacError_t registerMem(hostptr_t ptr, size_t size);
    gmacError_t unregisterMem(hostptr_t ptr);
#endif

    /* Synchronous interface */
    TESTABLE gmacError_t copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size, core::hpe::Mode &mode);
    TESTABLE gmacError_t copyToHost(hostptr_t host, const accptr_t acc, size_t size, core::hpe::Mode &mode);
    TESTABLE gmacError_t copyAccelerator(accptr_t dst, const accptr_t src, size_t size, stream_t stream);

    /* Asynchronous interface */
    TESTABLE gmacError_t copyToAcceleratorAsync(accptr_t acc, core::IOBuffer &buffer, size_t bufferOff, size_t count, core::hpe::Mode &mode, CUstream stream);
    TESTABLE gmacError_t copyToHostAsync(core::IOBuffer &buffer, size_t bufferOff, const accptr_t acc, size_t count, core::hpe::Mode &mode, CUstream stream);

    CUstream createCUstream();
    void destroyCUstream(CUstream stream);
    CUresult queryCUstream(CUstream stream);
    gmacError_t syncStream(CUstream stream);

    CUresult queryCUevent(CUevent event);
    gmacError_t syncCUevent(CUevent event);
    gmacError_t timeCUevents(uint64_t &t, CUevent start, CUevent end);

    gmacError_t execute(KernelLaunch &launch);

    gmacError_t memset(accptr_t addr, int c, size_t size, stream_t stream);

    gmacError_t sync();

    gmacError_t hostAlloc(hostptr_t &addr, size_t size, GmacProtection prot);
    gmacError_t hostFree(hostptr_t addr);
    accptr_t hostMap(const hostptr_t addr);

    static gmacError_t error(CUresult r);

    void getMemInfo(size_t &free, size_t &total) const;
    void getAcceleratorInfo(GmacAcceleratorInfo &info);

    bool hasUnifiedAddressing() const;
};

}}}

#include "Accelerator-impl.h"

#ifdef USE_DBC
#include "dbc/Accelerator.h"
#endif

#endif
