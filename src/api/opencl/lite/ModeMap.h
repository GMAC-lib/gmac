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

#ifndef GMAC_API_OPENCL_LITE_MODEMAP_H_
#define GMAC_API_OPENCL_LITE_MODEMAP_H_

#include "config/common.h"
#include "config/config.h"
#include "util/Lock.h"

#if defined(__APPLE__)
#   include <OpenCL/cl.h>
#else 
#   include <CL/cl.h>
#endif

#include <map>

namespace __impl { namespace opencl { namespace lite {

class Mode;

class GMAC_LOCAL ModeMap :
    public std::map<cl_context, Mode *>,
    public gmac::util::RWLock {
protected:
    typedef std::map<cl_context, Mode *> Parent;
public:
    ModeMap();
    virtual ~ModeMap();

    bool insert(cl_context ctx, Mode &mode);
    void remove(cl_context ctx);

    Mode *get(cl_context ctx) const; 
    Mode *owner(const hostptr_t addr, size_t size) const;
};


}}}

namespace __dbc { namespace opencl { namespace lite {
typedef __impl::opencl::lite::ModeMap ModeMap;
}}}


#include "ModeMap-impl.h"

#endif
