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

#ifndef GMAC_CORE_HPE_KERNEL_H_
#define GMAC_CORE_HPE_KERNEL_H_

#include <vector>

#include "config/common.h"
#include "include/gmac/types.h"

#include "memory/Manager.h"

#include "Descriptor.h"

namespace __impl { namespace core { namespace hpe {

class Mode;
class KernelLaunch;

typedef Descriptor<gmac_kernel_id_t> KernelDescriptor;

/**
 * GMAC Kernel abstraction
 */
class GMAC_LOCAL Kernel : public KernelDescriptor{
    DBC_FORCE_TEST(Kernel)

public:
    /**
     * Kernel constructor
     *
     * \param k A reference to a kernel descriptor
     */
    Kernel(const KernelDescriptor &k);

    virtual ~Kernel() {}
};

/**
 * GMAC descriptor for kernel execution
 */
class GMAC_LOCAL KernelLaunch {
protected:
    /** Execution mode where the kernel will be executed */
    Mode &mode_;
#ifdef DEBUG
    /** Kernel ID */
    gmac_kernel_id_t k_;

    /**
     * Default constructor
     * \param mode Execution mode where the kernel will be executed
     * \param k Identifier of the kernel to be execute
     */
    KernelLaunch(Mode &mode, gmac_kernel_id_t k);
#else
    /**
     * Default constructor
     * \param mode Execution mode where the kernel will be executed
     */
    KernelLaunch(Mode &mode);
#endif

    std::map<unsigned, std::list<memory::ObjectInfo>::iterator > paramToParamPtr_;
    std::list<memory::ObjectInfo> usedObjects_;

public:
    /**
     * Default destructor
     */
    virtual ~KernelLaunch() {};

    /**
     * Execute the kernel
     * \return Error code
     */
    virtual gmacError_t execute() = 0;

    /**
     * Get the execution mode associated to the kernel
     * \return Execution mode
     */
    Mode &getMode();

    /**
     * Adds a new object to the kernel launch
     *
     * \param ptr Address of the object to be added
     * \param index Index of the ptr in the parameter list
     * \param prot Access type of the object in the GPU
     */
    void addObject(hostptr_t ptr, unsigned index, GmacProtection prot);

    /**
     * Gets the list of objects being used by the kernel
     *
     * \param ptr Address of the object to be added
     */
    const std::list<memory::ObjectInfo> &getObjects() const;

#ifdef DEBUG
    /**
     * Get the execution kernel id
     * \return Kernel identified
     */
    gmac_kernel_id_t getKernelId() const;
#endif
};

}}}

#include "Kernel-impl.h"

#ifdef USE_DBC
#include "core/hpe/dbc/Kernel.h"
#endif

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
