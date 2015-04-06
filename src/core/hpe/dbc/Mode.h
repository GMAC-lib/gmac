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

#ifndef GMAC_CORE_HPE_DBC_MODE_H_
#define GMAC_CORE_HPE_DBC_MODE_H_

namespace __dbc { namespace core { namespace hpe {

class GMAC_LOCAL Mode :
    public __impl::core::hpe::Mode,
    public virtual Contract {
    DBC_TESTED(__impl::core::hpe::Mode)

private:
    typedef __impl::core::hpe::Mode Parent;

    typedef __impl::core::IOBuffer IOBufferImpl;
    typedef __impl::core::hpe::Accelerator AcceleratorImpl;
    typedef __impl::core::hpe::Kernel KernelImpl;
    typedef __impl::core::hpe::Process ProcessImpl;
    typedef __impl::core::hpe::AddressSpace AddressSpaceImpl;

protected:
    void cleanUpContexts();
    gmacError_t cleanUp();

public:
    Mode(ProcessImpl &proc, AcceleratorImpl &acc, AddressSpaceImpl &aSpace);
    virtual ~Mode();

    gmacError_t map(accptr_t &dst, hostptr_t src, size_t size, unsigned align = 1);
    gmacError_t unmap(hostptr_t addr, size_t size);
    gmacError_t copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size);
    gmacError_t copyToHost(hostptr_t host, const accptr_t acc, size_t size);
    gmacError_t copyAccelerator(accptr_t dst, const accptr_t src, size_t size);
    gmacError_t memset(accptr_t addr, int c, size_t size);
    gmacError_t bufferToAccelerator(accptr_t dst, IOBufferImpl &buffer, size_t size, size_t off = 0);
    gmacError_t acceleratorToBuffer(IOBufferImpl &buffer, const accptr_t dst, size_t size, size_t off = 0);
    void registerKernel(gmac_kernel_id_t k, KernelImpl &kernel);
    std::string getKernelName(gmac_kernel_id_t k) const;
    gmacError_t moveTo(AcceleratorImpl &acc);
};

}}}

#endif /* BLOCK_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
