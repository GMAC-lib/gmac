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

#ifndef GMAC_MEMORY_PROTOCOL_LAZY_BLOCKSTATE_H_
#define GMAC_MEMORY_PROTOCOL_LAZY_BLOCKSTATE_H_

#include "memory/Block.h"
#include "util/ReusableObject.h"

#include "memory/protocol/common/BlockState.h"

#include "LazyTypes.h"

namespace __impl {
namespace memory {

template <typename State> class StateBlock;

namespace protocol {
namespace lazy {

#if defined(USE_SUBBLOCK_TRACKING) || defined(USE_VM)
template <typename T>
class Array {
    T *array_;
    size_t size_;
public:
    explicit Array(size_t size) :
        size_(size)
    {
        array_ = new T[size];
    }

    ~Array()
    {
        delete [] array_;
    }

    T &operator[](const unsigned index)
    {
        ASSERTION(index < size_, "Index: %u. Size: "FMT_SIZE, index, size_);
        return array_[index];
    }

    const T &operator[](const unsigned index) const
    {
        ASSERTION(index < size_, "Index: %u. Size: "FMT_SIZE, index, size_);
        return array_[index];
    }

    size_t size() const
    {
        return size_;
    }
};

typedef Array<uint8_t> SubBlocks;
typedef Array<long_t> SubBlockCounters;

/// Tree used to group subblocks and speculatively unprotect them
struct GMAC_LOCAL BlockTreeState : public util::ReusableObject<BlockTreeState> {
    unsigned counter_;
    BlockTreeState();
};

class GMAC_LOCAL StrideInfo {
protected:
    lazy::Block &block_;

    unsigned stridedFaults_;
    long_t stride_;
    hostptr_t lastAddr_;
    hostptr_t firstAddr_;

public:
    StrideInfo(lazy::Block &block);

    void signalWrite(hostptr_t addr);

    unsigned getStridedFaults() const;
    bool isStrided() const;
    hostptr_t getFirstAddr() const;
    long_t getStride() const;

    void reset();
};


class GMAC_LOCAL BlockTreeInfo {
public:
    typedef std::pair<unsigned, unsigned> Pair;
protected:
    lazy::Block &block_;

    unsigned treeStateLevels_;
    BlockTreeState *treeState_;

    Pair lastUnprotectInfo_;

    Pair increment(unsigned subBlock);
public:
    BlockTreeInfo(lazy::Block &block);
    ~BlockTreeInfo();

    void signalWrite(const hostptr_t addr);
    Pair getUnprotectInfo();

    void reset();
};

#endif

class GMAC_LOCAL BlockState :
    public common::BlockState<lazy::State> {
#if defined(USE_SUBBLOCK_TRACKING)
    friend class StrideInfo;
    friend class BlockTreeInfo;
#endif

protected:
    lazy::Block &block();
    const lazy::Block &block() const;

#if defined(USE_SUBBLOCK_TRACKING)
    //const lazy::Block &block();
    unsigned subBlocks_;
    SubBlocks subBlockState_; 
#endif

#ifdef DEBUG
    // Global statistis
#if defined(USE_SUBBLOCK_TRACKING)
    SubBlockCounters subBlockFaultsRead_; 
    SubBlockCounters subBlockFaultsWrite_; 
    SubBlockCounters transfersToAccelerator_; 
    SubBlockCounters transfersToHost_; 
#else
    unsigned faultsRead_;
    unsigned faultsWrite_;
    unsigned transfersToAccelerator_;
    unsigned transfersToHost_;
#endif // USE_SUBBLOCK_TRACKING
#endif

#if defined(USE_SUBBLOCK_TRACKING)
    // Speculative subblock unprotect policies
    StrideInfo strideInfo_;
    BlockTreeInfo treeInfo_;

    void setSubBlock(const hostptr_t addr, ProtocolState state);
    void setSubBlock(long_t subBlock, ProtocolState state);
    void setAll(ProtocolState state);

    void reset();

    hostptr_t getSubBlockAddr(const hostptr_t addr) const;
    hostptr_t getSubBlockAddr(unsigned index) const;
    unsigned getSubBlocks() const;
    size_t getSubBlockSize() const;

    void writeStride(const hostptr_t addr);
    void writeTree(const hostptr_t addr);
#endif

public:
    BlockState(lazy::State init);

    void setState(ProtocolState state, hostptr_t addr = NULL);

#if 0
    bool hasState(ProtocolState state) const;
#endif

    gmacError_t syncToAccelerator();
    gmacError_t syncToHost();

    void read(const hostptr_t addr);
    void write(const hostptr_t addr);

    bool is(ProtocolState state) const;

    int protect(GmacProtection prot);
    int unprotect();

    void acquired();
    void released();

    gmacError_t dump(std::ostream &stream, common::Statistic stat);
};

}}}}

#include "BlockState-impl.h"

#endif // GMAC_MEMORY_PROTOCOL_LAZY_BLOCKSTATE_H_

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
