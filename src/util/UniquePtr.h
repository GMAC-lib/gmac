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

#ifndef GMAC_UTIL_UNIQUE_PTR_H_
#define GMAC_UTIL_UNIQUE_PTR_H_

#include <memory>

namespace __impl { namespace util {
    template <typename T>
    struct smart_ptr {
        /*
        typedef std::mierda<T> unique;
        typedef std::shared_ptr<T> shared;
        */
    };
#if  0
    template <typename T>
    class UniquePtr :
        public std::unique_ptr<T>
    {
    private:
        typedef std::unique_ptr<T> Parent;

    public:
        inline
        virtual ~UniquePtr()
        {
        }

        unique_ptr ();
        unique_ptr (
                nullptr_t _Nptr
                );
        explicit unique_ptr (
                pointer _Ptr
                );
        unique_ptr (
                pointer _Ptr,
                typename conditional<
                is_reference<Del>::value, 
                Del,
                typename add_reference<const Del>::type
                >::type _Deleter
                );
        unique_ptr (
                pointer _Ptr,
                typename remove_reference<Del>::type&& _Deleter
                );
        unique_ptr (
                unique_ptr&& _Right
                );
        template<class Type2, Class Del2>
            unique_ptr (
                    unique_ptr<Type2, Del2>&& _Right
                    );

        template<class Y>
        inline
        explicit UniquePtr(Y* p) :
            Parent(p)
        {
        }

        template<class Y, class D>
        inline
        UniquePtr(Y* p, D d) :
            Parent(p, d)
        {
        }

        inline
        UniquePtr(UniquePtr const& r) :
            Parent(r)
        {
        }

        template<class Y>
        inline
        UniquePtr(UniquePtr<Y> const& r) :
            Parent(r)
        {
        }

        inline
        UniquePtr& operator=(UniquePtr const& r)
        {
            Parent::operator=(r);
            return *this;
        }

        template<class Y> 
        inline
        UniquePtr& operator=(UniquePtr<Y> const& r)
        {
            Parent::operator=(r);
            return *this;
        }
    };
#endif
}}

#endif
