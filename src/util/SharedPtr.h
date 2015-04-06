/* Copyright (c) 2011 University of Illinois
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

#ifndef GMAC_UTIL_SHARED_PTR_H_
#define GMAC_UTIL_SHARED_PTR_H_

#include "config/common.h"

#if defined(POSIX)
#include <tr1/memory>
#elif defined(WINDOWS)
#include <memory>
#endif

namespace __impl { namespace util {
    template <typename T>
    class SharedPtr :
        public std::tr1::shared_ptr<T>
    {
    private:
        typedef std::tr1::shared_ptr<T> Parent;

    public:
        inline
        virtual ~SharedPtr()
        {
        }

        template<class Y>
        inline
        explicit SharedPtr(Y* p) :
            Parent(p)
        {
        }

        template<class Y, class D>
        inline
        SharedPtr(Y* p, D d) :
            Parent(p, d)
        {
        }

        inline
        SharedPtr(SharedPtr const& r) :
            Parent(r)
        {
        }

        template<class Y>
        inline
        SharedPtr(SharedPtr<Y> const& r) :
            Parent(r)
        {
        }

        inline
        SharedPtr& operator=(SharedPtr const& r)
        {
            Parent::operator=(r);
            return *this;
        }

        template<class Y> 
        inline
        SharedPtr& operator=(SharedPtr<Y> const& r)
        {
            Parent::operator=(r);
            return *this;
        }
    };

}}

#endif
