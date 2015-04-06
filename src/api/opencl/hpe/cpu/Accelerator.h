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
WITH THE SOFTWARE.
*/

#ifndef GMAC_API_OPENCL_HPE_CPU_ACCELERATOR_H_
#define GMAC_API_OPENCL_HPE_CPU_ACCELERATOR_H_

#include "api/opencl/hpe/Accelerator.h"

namespace __impl { namespace opencl { namespace hpe { namespace cpu {

/** An OpenCL capable accelerator */
class GMAC_LOCAL Accelerator :
    public gmac::opencl::hpe::Accelerator {

public:
    /** Default constructor
     * \param n Accelerator number
     * \param context OpenCL context the accelerator belongs to
     * \param device OpenCL device ID for the accelerator
     * \param major OpenCL major version supported by the accelerator
     * \param minor OpenCL minor version supported by the accelerator
     */
    Accelerator(int n, cl_context context, cl_device_id device, unsigned major, unsigned minor);
    /** Default destructor */
    virtual ~Accelerator();

    gmacError_t copyToAcceleratorAsync(accptr_t acc, core::IOBuffer &buffer, size_t bufferOff, size_t count, core::hpe::Mode &mode, cl_command_queue stream);
    gmacError_t copyToHostAsync(core::IOBuffer &buffer, size_t bufferOff, const accptr_t acc, size_t count, core::hpe::Mode &mode, cl_command_queue stream);
};

}}}}


#endif
