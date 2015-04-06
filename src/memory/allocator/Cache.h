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

#ifndef GMAC_MEMORY_ALLOCATOR_CACHE_H_
#define GMAC_MEMORY_ALLOCATOR_CACHE_H_

#include <list>
#include <map>

#include "config/common.h"
#include "memory/Manager.h"
#include "util/Lock.h"

namespace __impl { namespace memory { namespace allocator {

typedef std::list<hostptr_t> ObjectList;

/**
 * Arenas used by the caches of the slab allocator
 * \sa Cache
 * \sa Slab
 */
class GMAC_LOCAL Arena {
protected:
    hostptr_t ptr_;
    size_t size_;

    ObjectList objects_;

    Manager &manager_;
    core::Mode &mode_;

public:
    Arena(Manager &manager, core::Mode &mode, size_t objSize);
    ~Arena();

    inline hostptr_t address() const { return ptr_; }
    hostptr_t key() const;
    const ObjectList &objects() const;

    bool valid() const;
    bool full() const;
    bool empty() const;

    hostptr_t get();
    void put(hostptr_t obj);
};

/**
 * Caches, composed of arenas, used by the slab allocator
 * \sa Arena
 * \sa Slab
 */
class GMAC_LOCAL Cache :
    protected gmac::util::Lock {
protected:
    size_t objectSize;
    size_t arenaSize;

    typedef std::map<hostptr_t, Arena *> ArenaMap;
    ArenaMap arenas;

    Manager &manager_;
    core::Mode &mode_;
public:
    Cache(Manager &manager, core::Mode &mode, size_t size);
    virtual ~Cache();

    static Cache &get(long key, size_t size);
    static void cleanup();

    hostptr_t get();
    void put(hostptr_t obj);

};

}}}

#include "Cache-impl.h"

#endif
