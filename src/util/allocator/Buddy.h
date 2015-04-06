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

#ifndef GMAC_UTIL_ALLOCATOR_BUDDY_H_
#define GMAC_UTIL_ALLOCATOR_BUDDY_H_

#include <list>
#include <map>

#include "config/common.h"
#include "util/Lock.h"

namespace __impl { namespace util { namespace allocator {

/**
 * Simple buddy allocator
 */
class GMAC_LOCAL Buddy : protected gmac::util::Lock  {
    DBC_FORCE_TEST(__impl::util::allocator::Buddy)

protected:
    hostptr_t addr_;
    uint32_t size_;
    uint8_t index_;

    uint8_t ones(register uint32_t x) const;
    uint8_t index(register uint32_t x) const;
    uint32_t round(register uint32_t x) const;

    typedef std::list<off_t> List;
    typedef std::map<uint8_t, List> Tree;

    Tree _tree;
    TESTABLE off_t getFromList(uint8_t i);
    TESTABLE void putToList(off_t addr, uint8_t i);
public:
    Buddy(hostptr_t addr, size_t size);
    ~Buddy();

    inline hostptr_t addr() const { return addr_; }
    TESTABLE hostptr_t get(size_t &size);
    TESTABLE void put(hostptr_t addr, size_t size);
};

}}}

#if defined(USE_DBC)
#include "dbc/Buddy.h"
#endif

#endif /* BUDDY_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
