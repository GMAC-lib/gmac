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

#ifndef GMAC_API_OPENCL_IOBUFFER_H_
#define GMAC_API_OPENCL_IOBUFFER_H_

#if defined(__APPLE__)
#   include <OpenCL/cl.h>
#else 
#   include <CL/cl.h>
#endif

#include "api/opencl/Mode.h"
#include "api/opencl/Tracer.h"
#include "core/IOBuffer.h"
#include "util/ReusableObject.h"

namespace __impl { namespace opencl {

/**
 * IOBuffer implementation for OpenCL
 */
class GMAC_LOCAL IOBuffer :
    public gmac::core::IOBuffer,
    public __impl::util::ReusableObject<IOBuffer> {
protected:
    /** OpenCL buffer descriptor of the memory used by the buffer */
    cl_mem mem_;

    /** OpenCL event to query for the finalization of any ongoing data transfer */
    cl_event start_;

    /** OpenCL event to query for the finalization of any ongoing data transfer */
    cl_event event_;

    /** Execution mode using the I/O buffer */
    Mode *mode_;

    /** Signal is there are ongoing data transfers */
    bool started_;

    /** Last transfer size */
    size_t last_;

    /** Tracer */
    DataCommunication trace_;

public:
    /** Default constructor
     * \param mode Execution mode using the I/O buffer
     * \param addr Host memory address where to allocated the I/O buffer
     * \param size Size (in bytes) of the I/O buffer
     * \param mem cl_mem buffer
     * \param prot Tells whether the buffer is going to be read/written by the host
     * \return Error code
     */
    IOBuffer(Mode &mode, hostptr_t addr, size_t size, cl_mem mem, GmacProtection prot);

    /** Set the transfer direction from device to host
     * \param mode Execution mode performing the data transfer
     */
    void toHost(Mode &mode);

    /** Set the transfer direction from host to device
     * \param mode Execution mode performing the data transfer
     */
    void toAccelerator(Mode &mode);

    /** Set the event defining the start of a data transfer using the I/O buffer
     * \param event OpenCL event defining the starting time of a data transfer
     * \param size Size (in bytes) of the I/O transfer
     */
    void started(cl_event event, size_t size);

        /** Set the event defining the start of a data transfer using the I/O buffer
     * \param start OpenCL event defining the starting time of a data transfer
     * \param end OpenCL event defining the end of a data transfer
     * \param size Size (in bytes) of the I/O transfer
     */
    void started(cl_event start, cl_event end, size_t size);

    /** Waits for any incoming data transfers to finish
     * \return Error code
     */
    gmacError_t wait(bool internal = false);

    cl_mem getCLBuffer() { return mem_; }

    void setAddr(hostptr_t addr) { addr_ = addr; }
};

}}

#include "IOBuffer-impl.h"

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
