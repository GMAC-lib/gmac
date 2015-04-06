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

#ifndef GMAC_CORE_HPE_ADDRESS_SPACE_H_
#define GMAC_CORE_HPE_ADDRESS_SPACE_H_

#include <map>
#include <set>

#include "config/common.h"
#include "util/Lock.h"
#include "util/NonCopyable.h"

#include "memory/ObjectMap.h"

namespace __impl {


namespace memory {
class Object;
}


namespace core { namespace hpe {

class Mode;
class Process;

//! An object map associated to an execution mode
class GMAC_LOCAL AddressSpace :
    public memory::ObjectMap {

    typedef memory::ObjectMap Parent;

protected:
    /**
     * Execution mode owning this map
     */
    Process &parent_;

public:
    /**
     * Default constructor
     *
     * \param name Name of the object map used for tracing
     * \param parent Process that owns the map
     */
    AddressSpace(const char *name, Process &parent);

    /**
     * Default destructor
     */
    virtual ~AddressSpace();

    /**
     * Insert an object in the map and the global process map where all objects
     * are registered
     *
     * \param obj Object to remove from the map
     * \return True if the object was successfuly removed
     */
    bool addObject(memory::Object &obj);

    /**
     * Remove an object from the map and from the global process map where all
     * objects are registered
     *
     * \param obj Object to remove from the map
     * \return True if the object was successfuly removed
     */
    bool removeObject(memory::Object &obj);

    /**
     * Find the first object in a memory range in this map or on the global and
     * shared process object maps
     *
     * \param addr Starting address of the memory range where the object is
     * located
     * \param size Size (in bytes) of the memory range where the object is
     * located
     * \return First object within the memory range. NULL if no object is found
     */
    memory::Object *getObject(const hostptr_t addr, size_t size) const;

    /**
     * Add an owner to all global process objects
     *
     * \param proc Process whose global objects will be the owner added to
     * \param mode Owner to be added to global objects
     */
    static void addOwner(Process &proc, Mode &mode);

    /**
     * Remove an owner to all global process objects
     *
     * \param proc Process whose global objects will be the owner removed from
     * \param mode Owner to be removed from global objects
     */
    static void removeOwner(Process &proc, Mode &mode);
};

}}}

#endif
