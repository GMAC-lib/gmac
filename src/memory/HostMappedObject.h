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

#ifndef GMAC_MEMORY_HOSTMAPPEDOBJECT_H_
#define GMAC_MEMORY_HOSTMAPPEDOBJECT_H_

#include <map>

#include "config/common.h"

#include "util/Lock.h"
#include "util/Reference.h"

namespace __impl { 

namespace core {
	class Mode;
}

namespace memory {

class HostMappedObject;
//! A set of Host-mapped memory blocks
/*! This class is actually a map, because we need to easily locate blocks
    using host memory addresses
*/
class GMAC_LOCAL HostMappedSet : protected std::map<void *, HostMappedObject *>, 
    public gmac::util::RWLock {
protected:
    friend class HostMappedObject;

    typedef std::map<void *, HostMappedObject *> Parent;

    //! Inster a host mapped object in the set
    /*!
        \param object Host  mapped object to be inserted
        \return True if the block was inserted
    */
    bool insert(HostMappedObject *object);

    //! Find a host mapped object that contains a memory address
    /*!
        \param addr Host memory address within the host mapped object
        \return Host mapped object containing the memory address. NULL if not found
    */
    HostMappedObject *get(hostptr_t addr) const;
public:
    //! Default constructor
    HostMappedSet();

    //! Default destructor
    ~HostMappedSet();

    //! Remove a block from the list that contains a given host memory address
    /*!
        \param addr Memory address within the block to be removed
        \return True if an object was removed
    */
    bool remove(hostptr_t addr);
};

//! A memory object that only resides in host memory, but that is accessible from the accelerator
class GMAC_LOCAL HostMappedObject : public util::Reference {
protected:
    //! Starting host memory address of the object
    hostptr_t addr_;

    //! Size (in bytes) of the object
    size_t size_;

    //! Set of all host mapped memory objects
    static HostMappedSet set_;

    hostptr_t alloc(core::Mode &mode);
    void free(core::Mode &mode);
    accptr_t getAccPtr(core::Mode &mode) const;

    core::Mode &owner_;
public:
    /**
     * Default constructor
     * 
     * \param mode Execution mode creating the object
     * \param size Size (in bytes) of the object being created
     */
    HostMappedObject(core::Mode &mode, size_t size);

    /// Default destructor
    virtual ~HostMappedObject();

#ifdef USE_OPENCL
    gmacError_t acquire(core::Mode &current);
    gmacError_t release(core::Mode &current);
#endif
    
    /**
     * Get the starting host memory address of the object
     * 
     * \return Starting host memory address of the object
     */
    hostptr_t addr() const;

    /**
     * Get the size (in bytes) of the object
     * 
     * \return Size (in bytes) of the object
     */
    size_t size() const;

    /**
     * Get an accelerator address where a host address within the object can be accessed
     *
     * \param current Reference to the mode to which generate the address
     * \param addr Host memory address within the object
     * \return Accelerator memory address where the requested host memory address is mapped
     */
    accptr_t acceleratorAddr(core::Mode &current, const hostptr_t addr) const;

    /**
     * Remove a host mapped object from the list of all present host mapped object
     *
     * \param addr Host memory address within the object to be removed
     */
    static void remove(hostptr_t addr);

    /**
     * Get the host mapped memory object that contains a given host memory address
     *
     * \param addr Host memory address within the object
     * \return Host mapped object cotainig the host memory address. NULL if not found
     */
    static HostMappedObject *get(const hostptr_t addr);
};

}}

#include "HostMappedObject-impl.h"


#endif
