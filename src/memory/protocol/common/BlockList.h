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

#ifndef GMAC_MEMORY_PROTOCOL_BLOCKLIST_H_
#define GMAC_MEMORY_PROTOCOL_BLOCKLIST_H_

#include <list>
#include <map>
#include <vector>

#include "config/common.h"
#include "include/gmac/types.h"
#include "util/Lock.h"

namespace __impl {

namespace core {
    class Mode;
}

namespace memory {
class Block;

namespace protocol {

//! FIFO list of blocks
class GMAC_LOCAL BlockList :
    protected std::list<Block *>,
    public gmac::util::SpinLock {
// We need a locked list becase execution modes might be shared among different threads
protected:
    typedef std::list<Block *> Parent;

public:
    /// Default constructor
    BlockList();

    /// Default destructor
    virtual ~BlockList();

    /** Whether the list is empty or not
     *
     * \return True if the list is empty
     */
    bool empty() const;

    /** Size of the list
     *
     *  \return Number of blocks in the list
     */
    size_t size() const;

    /** Add a block to the end of list
     *
     * \param block Block to be addded to the end of list
     */
    void push(Block &block);

    /** Return the first Block in the list
     *
     * \return Block from extracted from the begining of the list
     */
    Block &front();

    /** Remove a block from the list
     *
     * \param block Block to be removed from the list
     */
    void remove(Block &block);
};

}}}

#include "BlockList-impl.h"

#endif
