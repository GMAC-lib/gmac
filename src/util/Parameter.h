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

#ifndef GMAC_UTIL_PARAMETER_H_
#define GMAC_UTIL_PARAMETER_H_

#include <map>
#include <iostream>
#include <vector>

#include "config/common.h"

enum GMAC_LOCAL ParamFlags {
    PARAM_NONZERO = 0x1
};

namespace __impl { namespace util {
//! Base abstract class to allow generic calls
class GMAC_LOCAL __Parameter {
public:
    //! Default destructor
    virtual ~__Parameter() {}

    //! Print the parameter name and value in the screen
    virtual void print() const = 0;
};

//! A parameter whose value is taken from environment variables
template<typename T>
class GMAC_LOCAL Parameter : public __Parameter {
protected:
    //! Parameter value
    T *value_;

    //! Default parameter value
    T def_;

    //! Parameter name
    const char *name_;

    //! Environment variable to get the parameter from
    const char *envVar_;

    //! Parameter flags (e.g. not null value)
    uint32_t flags_;

    //! Wheather there is an environment variable setting this parameter or not
    bool envSet_;

public:
    //! Default destructor
    virtual ~Parameter() {}

    //! Default constructor
    /*!
        \param address 
        \param name Parameter name
        \param def Default parameter value
        \param envVar Environment variable that sets this parameter
        \param flags Parameter flags
    */
    Parameter(T *address, const char *name, T def, const char *envVar,
        uint32_t flags = 0);

    void print() const;
};

namespace params {
typedef struct GMAC_LOCAL {
    __impl::util::__Parameter *(*ctor)(void);
    __impl::util::__Parameter *param;
} ParameterCtor;

extern ParameterCtor ParamCtorList[];

#define PARAM(v, t, d, ...)  extern t v;

#include "util/Parameter-def.h"
}

}}

#include "Parameter-impl.h"

#endif
