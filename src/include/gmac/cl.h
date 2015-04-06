/* Copyright (c) 2009, 2011 University of Illinois
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

#ifndef GMAC_CL_H_
#define GMAC_CL_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

#if defined(__APPLE__)
#   include <OpenCL/cl.h>
#else
#   include <CL/cl.h>
#endif
#include "cl_types.h"
#include "visibility.h"

/**
 *  Allocates a OpenCL memory buffer accessible from the host
 *  \param queue OpenCL command queue attached to the context where the object will be allocated
 *  \param addr Reference to the host memory address where the data will be accessible
 *  \param count Size (in bytes) of the data to be allocated
 *  \return OpenCL error code
 */
GMAC_API cl_int APICALL clMalloc(cl_command_queue queue, void **addr, size_t count);


/**
 *  Release the OpenCL buffer associated to a host memory address
 *  \param queue OpenCL command queue attached to the context where the object was allocated
 *  \param addr Host memory address
 *  \return OpenCL error code
 */
GMAC_API cl_int APICALL clFree(cl_command_queue queue, void *addr);


/**
 *  Returns the OpenCL buffer associated to a host memory address
 *  \param context OpenCL context where the obejct was allocated
 *  \param addr Host memory address
 *  \return OpenCL error code
 */
GMAC_API cl_mem APICALL clGetBuffer(cl_context context, const void *addr);

/**
 * Initialize the OpenCL/Helper states
 * \param platforms A pointer
 * \return OpenCL error code
 */
GMAC_API cl_int APICALL clInitHelpers(size_t *platforms);

/**
 * Gets the OpenCL/Helper states for each context
 * \return An array of helpers (one helper per platform)
 */
GMAC_API cl_helper * APICALL clGetHelpers();

/**
 * Release the OpenCL/Helpers' state
 * \return OpenCL error code
 */
GMAC_API cl_int APICALL clReleaseHelpers();

/**
 * Load a set of kernels from a file
 * \param state OpenCL/Helper state where the OpenCL program is loaded
 * \param path String that contains the path to the file with the OpenCL code
 * \return OpenCL error code
 */
GMAC_API cl_int APICALL clHelperLoadProgramFromFile(cl_helper state, const char *path);


#ifdef __cplusplus
}

template<typename T>
static inline cl_mem clGetBuffer(cl_context context, const T *addr)
{
    return clGetBuffer(context, (const void *)addr);
}

#endif

#undef __dv

#endif /* OPENCL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
