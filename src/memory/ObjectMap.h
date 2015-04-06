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

#ifndef GMAC_MEMORY_MAP_H_
#define GMAC_MEMORY_MAP_H_

#include <map>
#include <set>

#include "config/common.h"
#include "util/Lock.h"
#include "util/NonCopyable.h"

#include "protocol/common/BlockState.h"

namespace __impl {

namespace core {
class Mode;
class Process;
namespace hpe { class AddressSpace; }
}

namespace memory {
class Object;
class Protocol;

//! A map of objects that is not bound to any Mode
class GMAC_LOCAL ObjectMap :
     protected gmac::util::RWLock,
     protected std::map<const hostptr_t, Object *>,
     public util::NonCopyable {
protected:
    friend class core::hpe::AddressSpace;
    typedef std::map<const hostptr_t, Object *> Parent;

    Protocol &protocol_;

    bool modifiedObjects_;
    bool releasedObjects_;

#ifdef USE_VM
    __impl::memory::vm::Bitmap bitmap_;
#endif

#ifdef DEBUG
    static Atomic StatsInit_;
    static Atomic StatDumps_;
    static std::string StatsDir_;
    static void statsInit();
#endif

    void modifiedObjects_unlocked();

    /**
     * Find an object in the map
     *
     * \param addr Starting memory address within the object to be found
     * \param size Size (in bytes) of the memory range where the object can be
     * found
     * \return First object inside the memory range. NULL if no object is found
     */
    Object *mapFind(const hostptr_t addr, size_t size) const;
public:
    /**
     * Default constructor
     *
     * \param name Name of the object map used for tracing
     */
    ObjectMap(const char *name);

    /**
     * Default destructor
     */
    virtual ~ObjectMap();

    /**
     * Decrements the reference count of the contained objects
     */
    void cleanUp();


    /**
     * Get the number of objects in the map
     *
     * \return Number of objects in the map
     */
    size_t size() const;

    /**
     * Insert an object in the map
     *
     * \param obj Object to insert in the map
     * \return True if the object was successfuly inserted
     */
    virtual bool addObject(Object &obj);

    /**
     * Remove an object from the map
     *
     * \param obj Object to remove from the map
     * \return True if the object was successfuly removed
     */
    virtual bool removeObject(Object &obj);

    /**
     * Tells if an object belongs to the map
     *
     * \param obj Object to be checked
     * \return True if the object belongs to the map
     */
    bool hasObject(Object &obj) const;

    /**
     * Find the firs object in a memory range
     *
     * \param addr Starting address of the memory range where the object is
     * located
     * \param size Size (in bytes) of the memory range where the object is
     * located
     * \return First object within the memory range. NULL if no object is found
     */
    virtual Object *getObject(const hostptr_t addr, size_t size = 0) const;

    /**
     * Get the amount of memory consumed by all objects in the map
     *
     * \return Size (in bytes) of the memory consumed by all objects in the map
     */
    size_t memorySize() const;

    /**
     * Execute an operation on all the objects in the map
     *
     * \param f Operation to be executed
     * \sa __impl::memory::Object::acquire
     * \sa __impl::memory::Object::toHost
     * \sa __impl::memory::Object::toAccelerator
     * \return Error code
     */
    gmacError_t forEachObject(gmacError_t (Object::*f)(void));

    /**
     * Execute an operation on all the objects in the map passing an argument
     * \param f Operation to be executed
     * \param p Parameter to be passed
     * \sa __impl::memory::Object::removeOwner
     * \sa __impl::memory::Object::realloc
     * \return Error code
     */
    template <typename T>
    gmacError_t forEachObject(gmacError_t (Object::*f)(T &), T &p);

#ifdef DEBUG
    gmacError_t dumpObjects(const std::string &dir, std::string prefix, protocol::common::Statistic stat) const;
    gmacError_t dumpObject(const std::string &dir, std::string prefix, protocol::common::Statistic stat, hostptr_t ptr) const;
#endif

    /**
     * Tells if the objects of the mode have been already invalidated
     * \return Boolean that tells if objects of the mode have been already
     * invalidated
     */
    bool hasModifiedObjects() const;

    /**
     * Notifies the mode that one (or several) of its objects have been validated
     */
    void modifiedObjects();

    /**
     * Notifies the mode that one (or several) of its objects has been invalidated
     */
    void invalidateObjects();

    /**
     * Tells if the objects of the mode have been already released to the
     * accelerator
     * \return Boolean that tells if objects of the mode have been already
     * released to the accelerator
     */
    bool releasedObjects() const;

    /**
     * Releases the ownership of the objects of the mode to the accelerator
     * and waits for pending transfers
     */
    gmacError_t releaseObjects();

    /**
     * Waits for kernel execution and acquires the ownership of the objects
     * of the mode from the accelerator
     */
    gmacError_t acquireObjects();

    /**
     * Gets a reference to the memory protocol used by the mode
     * \return A reference to the memory protocol used by the mode
     */
    Protocol &getProtocol();

#ifdef USE_VM
    memory::vm::Bitmap &getBitmap();
    const memory::vm::Bitmap &getBitmap() const;
#endif
};

}}

#include "ObjectMap-impl.h"

#endif
