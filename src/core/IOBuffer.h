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

/**
 * \file core/IOBuffer.h
 *
 * I/O Buffer base abstraction
 */

#ifndef GMAC_CORE_IOBUFFER_H_
#define GMAC_CORE_IOBUFFER_H_

#include "config/common.h"
#include "include/gmac/types.h"
#include "util/Lock.h"
#include "util/NonCopyable.h"

namespace __impl { namespace core {

/**
 * Buffer used to optimize memory transfers
 */
class GMAC_LOCAL IOBuffer :
    public util::NonCopyable {
    DBC_FORCE_TEST(__impl::core::IOBuffer)

public:
    enum State { Idle, ToHost, ToAccelerator };

protected:
    /**
     * Address of the buffer's memory
     */
    void *addr_;

    /**
     * Size of the buffer
     */
    size_t size_;

    /**
     * Tells whether the buffer can be used in asynchronous transfers
     */
    bool async_;

    /**
     * Status of the memory transfers performed on the buffer
     */
    State state_;

    /**
     * Type of access performed on the buffer's memory
     */
    GmacProtection prot_;

    /**
     * Constructor of the buffer
     *
     * \param addr A pointer to the address to be used by the buffer
     * \param size The size of the buffer
     * \param async Indicates if the buffer can be used in asynchronous
     * transfers
     * \param prot Tells if the buffer is going to be read or written in the host
     */
    IOBuffer(void *addr, size_t size, bool async, GmacProtection prot);
public:
    /**
     * Destructor of the buffer
     */
    virtual ~IOBuffer();

    /**
     * Returns the starting address of the buffer memory
     *
     * \return The starting address of the buffer memory
     */
    TESTABLE uint8_t *addr() const;

    /**
     * Returns the end address of the buffer memory
     *
     * \return The end address of the buffer memory
     */
    TESTABLE uint8_t *end() const;

    /**
     * Returns the size in bytes of the buffer
     *
     * \return The size in bytes of the buffer
     */
    TESTABLE size_t size() const;

    /**
     * Tells if the buffer can be used in asynchronous memory transfers
     *
     * \return true if the buffer can be used in asynchronous memory transfers.
     * false otherwise
     */
    bool async() const;

    /**
     * Returns the current state of the buffer
     *
     * \return The current state of the buffer
     * \sa State
     */
    State state() const;

    /**
     * Waits for the transfer that uses the buffer
     *
     * \param internal Tells whether the function has been called within the execution layer or not
     * \return gmacSuccess on success. An error code otherwise
     */
    virtual gmacError_t wait(bool internal = false) = 0;

    GmacProtection getProtection() const;
};

}}

#include "IOBuffer-impl.h"

#ifdef USE_DBC
#include "core/dbc/IOBuffer.h"
#endif

#endif




/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
