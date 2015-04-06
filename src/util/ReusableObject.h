/* Copyright (c) 2009 University of Illinois
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


#ifndef GMAC_UTIL_REUSABLEOBJECT_H_
#define GMAC_UTIL_REUSABLEOBJECT_H_

#include <cstddef>

#include "config/common.h"

#include "Lock.h"

namespace __impl { namespace util {

template <typename T>
class GMAC_LOCAL Pool :
    protected gmac::util::Lock {
public:
    union Object {
        char dummy[sizeof(T)];
        Object * next;
    };

    Pool() :
        gmac::util::Lock("ReusableObjectPool"),
        freeList_(NULL)
    {}

    ~Pool()
    {
        Object * next, * tmp;
        for (next = freeList_; next; next = tmp) {
            tmp = next->next;
            delete next;
        }
    }

    T *get()
    {
        lock();
        Object *ret = freeList_;
        if (ret) freeList_ = ret->next;
        unlock();
        return (T *) ret;
    }

    void put(T *ptr)
    {
        lock();
        ((Object *) ptr)->next = freeList_;
        freeList_ = (Object *) ptr;
        unlock();
    }


private:
    Object * freeList_;
};

template <typename T>
class GMAC_LOCAL ReusableObject {
public:
    void *operator new(size_t bytes)
    {
        T *res = pool.get();
        return res? res : (T *) new typename Pool<T>::Object;
    }

    void operator delete(void *ptr)
    {
        pool.put((T *) ptr);
    }

protected:
    static Pool<T> pool;
};

// Static initialization

template<class T> GMAC_LOCAL Pool<T> 
ReusableObject<T>::pool;

}}

#endif /* CYCLE_UTILS_REUSABLE_OBJECT_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=100 foldmethod=marker expandtab cindent cinoptions=p5,t0,(0: */
