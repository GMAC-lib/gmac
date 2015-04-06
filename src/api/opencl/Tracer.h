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

#ifndef GMAC_API_OPENCL_TRACER_H_
#define GMAC_API_OPENCL_TRACER_H_

#include "trace/Tracer.h"


namespace __impl {


namespace opencl {

class Mode;

class DataCommunication {
protected:
#if defined(USE_TRACE)
    uint64_t stamp_;
    THREAD_T src_, dst_;
#endif
public:
    /** Default constructor */
    DataCommunication();

    /**
     * Initialize a data communication that will be traced later on
     * \param src Source thread
     * \param dst Destination thread
     */
    void init(THREAD_T src, THREAD_T dst);

    /** Get the curren thread ID
     * \return Thread ID
     */
    THREAD_T getThreadId() const;

    /**
     * Get the thread ID associated to a mode
     * \param mode Mode to get the ID from
     * \return Thread ID
     */
    THREAD_T getModeId(const Mode &mode) const;

    /**
     * Trace a data communication
     * \param start OpenCL event starting the communication
     * \param end OpenCL event ending the communication
     * \param size Size (in bytes) transferred
     */
    void trace(cl_event start, cl_event end, size_t size) const;
};

class KernelExecution {
protected:
#if defined(USE_TRACE)
    uint64_t stamp_;
    THREAD_T thread_;

    unsigned major_;
    unsigned minor_;

    struct TracePoint {
        uint64_t stamp;
        THREAD_T thread;
        static const size_t NameSize = 512;
        char name[NameSize];
    };
#if defined(_MSC_VER)
#	define STDCALL __stdcall
#else
#	define STDCALL
#endif
    static void STDCALL Callback(cl_event event, cl_int status, void *data);
#endif
public:
    /** Default constructor */
    KernelExecution(unsigned majort, unsigned minor);

    /**
     * Initialize a kernel execution that will be traced later on
     * \param thread GPU where the kernel is executed
     */
    void init(THREAD_T thread);

    /**
     * Get the thread ID associated to a mode
     * \param mode Mode to get the ID from
     * \return Thread ID
     */
    THREAD_T getModeId(const Mode &mode) const;

    /**
     * Trace a data communication
     * \param kernel OpenCL kernel handler to be traced
     * \param event OpenCL event starting the communication
     */
    void trace(cl_kernel kernel, cl_event event) const;

};

}}

#include "Tracer-impl.h"

#endif
