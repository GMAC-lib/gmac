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

#ifndef GMAC_MEMORY_PROTOCOL_LAZY_BLOCKSTATE_IMPL_H_
#define GMAC_MEMORY_PROTOCOL_LAZY_BLOCKSTATE_IMPL_H_

#include "memory/StateBlock.h"

#include <sstream>

#if defined(USE_SUBBLOCK_TRACKING) || defined(USE_VM)

#ifdef USE_VM
#include "core/hpe/Mode.h"
#include "core/hpe/Process.h"
#endif

#include "memory/Memory.h"
#include "memory/vm/Model.h"

namespace __impl {
namespace memory {
namespace protocol {
namespace lazy {

inline
BlockTreeState::BlockTreeState() :
    counter_(0)
{
}

inline
BlockTreeInfo::BlockTreeInfo(lazy::Block &block) :
    block_(block)
{ 
    // Initialize subblock state tree (for random access patterns)
    treeState_ = new BlockTreeState[2 * block.getSubBlocks() - 1];
    bool isPower;
    unsigned levels = log2(block.getSubBlocks(), isPower) + 1;
    if (isPower == false) {
        levels++;
    }
    treeStateLevels_ = levels;
}

inline
BlockTreeInfo::~BlockTreeInfo()
{
    delete treeState_;
}

inline
void
BlockTreeInfo::reset()
{
    ::memset(treeState_, 0, sizeof(BlockTreeState) * block_.getSubBlocks());
}

inline
BlockTreeInfo::Pair
BlockTreeInfo::increment(unsigned subBlock)
{
    unsigned level     = treeStateLevels_ - 1;
    unsigned levelOff  = 0;
    unsigned levelPos  = subBlock;
    unsigned levelSize = block_.getSubBlocks();
    unsigned children  = 1;
    unsigned inc       = 1;

    Pair ret;
    do {
        // printf("TREE! Level: %u (%u + %u)\n", level, levelOff, levelPos);
        unsigned idx = levelOff + levelPos;
        unsigned &counter = treeState_[idx].counter_;

        // printf("TREE! Counter(pre): %u\n", counter);
        if (counter == children) {
            //FATAL("Inconsistent state");
        } else if (2 * (counter + inc) > children) {
            inc     = children - counter;
            counter = children;

            ret.first  = subBlock & (~(children - 1));
            ret.second = children;
            if (ret.first + ret.second > block_.getSubBlocks()) {
                ret.second = block_.getSubBlocks() - ret.first;
            }
        } else {
            counter += inc;
        }
        // printf("TREE! Counter(post): %u\n", counter);

        levelOff  += levelSize;
        levelSize /= 2;
        levelPos  /= 2;
        children  *= 2;
    } while (level-- > 0);

    return ret;
}

inline
void
BlockTreeInfo::signalWrite(const hostptr_t addr)
{
    long_t currentSubBlock = GetSubBlockIndex(block_.addr(), addr);

    //printf("TREE! <%p> %u\n", block_.addr(), unsigned(currentSubBlock));
    lastUnprotectInfo_ = increment(currentSubBlock);
    //printf("TREE! Result: %u:%u\n", lastUnprotectInfo_.first, lastUnprotectInfo_.second);
}

inline
BlockTreeInfo::Pair
BlockTreeInfo::getUnprotectInfo() 
{
    return lastUnprotectInfo_;
}

inline
StrideInfo::StrideInfo(lazy::Block &block) :
    block_(block)
{
    reset();
}

inline unsigned
StrideInfo::getStridedFaults() const
{
    return stridedFaults_;
}

#define STRIDE_THRESHOLD 4

inline bool
StrideInfo::isStrided() const
{
    /// \todo Do not hardcode the threshold value
    return stridedFaults_ > STRIDE_THRESHOLD;
}

inline hostptr_t
StrideInfo::getFirstAddr() const
{
    return firstAddr_;
}

inline long_t
StrideInfo::getStride() const
{
    return stride_;
}

inline
void
StrideInfo::reset()
{
    stridedFaults_ = 0;
    stride_ = 0;
}

inline void
StrideInfo::signalWrite(hostptr_t addr)
{
    if (stridedFaults_ == 0) {
        stridedFaults_ = 1;
        firstAddr_ = addr;
        //printf("STRIDE 1\n");
    } else if (stridedFaults_ == 1) {
        stride_ = addr - lastAddr_;
        stridedFaults_ = 2;
        //printf("STRIDE 2: %lu\n", stride_);
    } else {
        if (addr == lastAddr_ + stride_) {
            stridedFaults_++;
            //printf("STRIDE 3a\n");
        } else {
            stride_ = addr - lastAddr_;
            stridedFaults_ = 2;
            //printf("STRIDE 3b: %lu\n", stride_);
        }
    }
    lastAddr_ = addr;
}

inline
Block &
BlockState::block()
{
	//return reinterpret_cast<Block &>(*this);
	return *(Block *)this;
}

inline
const Block &
BlockState::block() const
{
	//return reinterpret_cast<const Block &>(*this);
	return *(const Block *)this;
}

inline
void
BlockState::setSubBlock(const hostptr_t addr, ProtocolState state)
{
    setSubBlock(GetSubBlockIndex(block().addr(), addr), state);
}

#ifdef USE_VM
#define GetMode() (*(core::hpe::Mode *)(void *) &block_.owner(getProcess().getCurrentMode()))
#define GetBitmap() GetMode().getBitmap()
#endif

inline
void
BlockState::setSubBlock(long_t subBlock, ProtocolState state)
{
#ifdef USE_VM
    vm::Bitmap &bitmap = GetBitmap();
    bitmap.setEntry(block().acceleratorAddr(GetMode(), block().addr() + subBlock * SubBlockSize_), state);
#else
    subBlockState_[subBlock] = uint8_t(state);
#endif
}

inline
void
BlockState::setAll(ProtocolState state)
{
#ifdef USE_VM
    vm::Bitmap &bitmap = GetBitmap();
    bitmap.setEntryRange(block().acceleratorAddr(GetMode(), block().addr()), block().size(), state);
#else
    ::memset(&subBlockState_[0], uint8_t(state), subBlockState_.size() * sizeof(uint8_t));
#endif
}

inline
BlockState::BlockState(lazy::State init) :
    common::BlockState<lazy::State>(init),
    subBlocks_(block().size()/SubBlockSize_ + ((block().size() % SubBlockSize_ == 0)? 0: 1)),
    subBlockState_(subBlocks_),
#ifdef DEBUG
    subBlockFaultsRead_(subBlocks_),
    subBlockFaultsWrite_(subBlocks_),
    transfersToAccelerator_(subBlocks_),
    transfersToHost_(subBlocks_),
#endif
    strideInfo_(block()),
    treeInfo_(block()),
    faultsRead_(0),
    faultsWrite_(0)
{ 
    // Initialize subblock states
#ifndef USE_VM
    setAll(init);
#endif

#ifdef DEBUG
    ::memset(&subBlockFaultsRead_[0],     0, subBlockFaultsRead_.size() * sizeof(long_t));
    ::memset(&subBlockFaultsWrite_[0],    0, subBlockFaultsWrite_.size() * sizeof(long_t));
    ::memset(&transfersToAccelerator_[0], 0, transfersToAccelerator_.size() * sizeof(long_t));
    ::memset(&transfersToHost_[0],        0, transfersToHost_.size() * sizeof(long_t));
#endif
}

#if 0
common::BlockState<lazy::State>::ProtocolState
BlockState::getState(hostptr_t addr) const
{
    return ProtocolState(subBlockState_[GetSubBlockIndex(block().addr(), addr)]);
}
#endif

inline void
BlockState::setState(ProtocolState state, hostptr_t addr)
{
    if (addr == NULL) {
        setAll(state);
        state_ = state;
    } else {
        if (state == lazy::Dirty) {
            state_ = lazy::Dirty;
        } else if (state == lazy::ReadOnly) {
            if (state_ != lazy::Dirty) state_ = lazy::ReadOnly;
        } else {
            FATAL("Wrong state transition");
        }

        subBlockState_[GetSubBlockIndex(block().addr(), addr)] = state;
    }
}

inline hostptr_t
BlockState::getSubBlockAddr(const hostptr_t addr) const
{
    return GetSubBlockAddr(block().addr(), addr);
}

inline hostptr_t
BlockState::getSubBlockAddr(unsigned index) const
{
    return GetSubBlockAddr(block().addr(), block().addr() + index * SubBlockSize_);
}

inline unsigned
BlockState::getSubBlocks() const
{
    return subBlocks_;
}

inline size_t
BlockState::getSubBlockSize() const
{
    return block().size() < SubBlockSize_? block().size(): SubBlockSize_;
}

inline gmacError_t
BlockState::syncToAccelerator()
{
    gmacError_t ret = gmacSuccess;

    unsigned groupStart = 0, groupEnd = 0;
    unsigned gaps = 0;
    bool inGroup = false;

    TRACE(LOCAL, "Transfer block to accelerator: %p", block().addr());

#ifdef USE_VM
    vm::Bitmap &bitmap = GetBitmap();
#endif
    for (unsigned i = 0; i != subBlocks_; i++) {
#ifdef USE_VM
        if (bitmap.getEntry<ProtocolState>(block().acceleratorAddr(GetMode()) + i * SubBlockSize_) == lazy::Dirty) {
#else
        if (subBlockState_[i] == lazy::Dirty) {
#endif
            if (!inGroup) {
                groupStart = i;
                inGroup = true;
            }
            setSubBlock(i, lazy::ReadOnly);
            groupEnd = i;
        } else if (inGroup) {
            if (vm::costGaps<vm::MODEL_TODEVICE>(SubBlockSize_, gaps + 1, i - groupStart + 1) <
                    vm::cost<vm::MODEL_TODEVICE>(SubBlockSize_, 1)) {
                gaps++;
            } else {
                size_t sizeTransfer = SubBlockSize_ * (groupEnd - groupStart + 1);
                if (sizeTransfer > block().size()) sizeTransfer = block().size();
                ret = block().toAccelerator(groupStart * SubBlockSize_, sizeTransfer);
#ifdef DEBUG
                for (unsigned j = groupStart; j <= groupEnd; j++) { 
                    transfersToAccelerator_[j]++;
                }
#endif
                gaps = 0;
                inGroup = false;
                if (ret != gmacSuccess) break;
            }
        }
    }

    if (inGroup) {
        size_t sizeTransfer = SubBlockSize_ * (groupEnd - groupStart + 1);
        if (sizeTransfer > block().size()) sizeTransfer = block().size();
        ret = block().toAccelerator(groupStart * SubBlockSize_, sizeTransfer);
                                    
#ifdef DEBUG
        for (unsigned j = groupStart; j <= groupEnd; j++) { 
            transfersToAccelerator_[j]++;
        }
#endif

    }

	return ret;
}

inline gmacError_t
BlockState::syncToHost()
{
    TRACE(LOCAL, "Transfer block to host: %p", block().addr());

#ifndef USE_VM
    gmacError_t ret = block().toHost();
#ifdef DEBUG
    for (unsigned i = 0; i < subBlockState_.size(); i++) { 
        transfersToHost_[i]++;
    }
#endif

#else
    gmacError_t ret = gmacSuccess;

    unsigned groupStart = 0, groupEnd = 0;
    unsigned gaps = 0;
    bool inGroup = false;

    vm::Bitmap &bitmap = GetBitmap();
    for (unsigned i = 0; i != subBlocks_; i++) {
        if (bitmap.getEntry<ProtocolState>(block().acceleratorAddr(GetMode()) + i * SubBlockSize_) == lazy::Invalid) {
#ifdef DEBUG
            transfersToHost_[i]++;
#endif
            if (!inGroup) {
                groupStart = i;
                inGroup = true;
            }
            setSubBlock(i, lazy::ReadOnly);
            groupEnd = i;
        } else if (inGroup) {
            if (vm::costGaps<vm::MODEL_TOHOST>(SubBlockSize_, gaps + 1, i - groupStart + 1) <
                    vm::cost<vm::MODEL_TOHOST>(SubBlockSize_, 1)) {
                gaps++;
            } else {
                ret = block().toHost(groupStart * SubBlockSize_, SubBlockSize_ * (groupEnd - groupStart + 1) );
                gaps = 0;
                inGroup = false;
                if (ret != gmacSuccess) break;
            }
        }
    }

    if (inGroup) {
        ret = block().toHost(groupStart * SubBlockSize_,
                            SubBlockSize_ * (groupEnd - groupStart + 1));
    }

#endif

    return ret;
}

inline void
BlockState::read(const hostptr_t addr)
{
    long_t currentSubBlock = GetSubBlockIndex(block().addr(), addr);
    faultsRead_++;
    faultsCacheRead_++;

    setSubBlock(currentSubBlock, lazy::ReadOnly);
#ifdef DEBUG
    subBlockFaultsRead_[currentSubBlock]++;
#endif
    TRACE(LOCAL, "");

    return;
}

inline void
BlockState::writeStride(const hostptr_t addr)
{
    strideInfo_.signalWrite(addr);
    if (strideInfo_.isStrided()) {
        for (hostptr_t cur = strideInfo_.getFirstAddr(); cur >= block().addr() &&
                cur < (block().addr() + block().size());
                cur += strideInfo_.getStride()) {
            long_t subBlock = GetSubBlockIndex(block().addr(), cur);
            setSubBlock(subBlock, lazy::Dirty);
        }
    }
}

inline void
BlockState::writeTree(const hostptr_t addr)
{
    treeInfo_.signalWrite(addr);
    BlockTreeInfo::Pair info = treeInfo_.getUnprotectInfo();

    for (unsigned i = info.first; i < info.first + info.second; i++) {
        setSubBlock(i, lazy::Dirty);
    }
}

inline void
BlockState::write(const hostptr_t addr)
{
    long_t currentSubBlock = GetSubBlockIndex(block().addr(), addr);

    faultsWrite_++;
    faultsCacheWrite_++;

    setSubBlock(currentSubBlock, lazy::Dirty);

#ifdef DEBUG
    subBlockFaultsWrite_[currentSubBlock]++;
#endif

    if (subBlockState_.size() > STRIDE_THRESHOLD) {
        if (util::params::ParamSubBlockStride) {
            writeStride(addr);
            if (util::params::ParamSubBlockTree && !strideInfo_.isStrided()) {
                writeTree(addr);
            }
        } else if (util::params::ParamSubBlockTree) {
            writeTree(addr);
        }
    }
}

inline bool
BlockState::is(ProtocolState state) const
{
#ifdef USE_VM
    vm::Bitmap &bitmap = GetBitmap();
    return bitmap.isAnyInRange(block().acceleratorAddr(GetMode(), block().addr()), block().size(), state);
#else
    for (unsigned i = 0; i < subBlockState_.size(); i++) {
        if (subBlockState_[i] == state) return true;
    }

    return false;
#endif
}

inline void
BlockState::reset()
{
    faultsRead_  = 0;
    faultsWrite_ = 0;

    if (util::params::ParamSubBlockStride) strideInfo_.reset();
    if (util::params::ParamSubBlockTree) treeInfo_.reset();
}

inline int
BlockState::unprotect()
{
    int ret = 0;
    unsigned start = 0;
    unsigned size = 0;
#ifdef USE_VM
    vm::Bitmap &bitmap = GetBitmap();
#endif
    for (unsigned i = 0; i < subBlocks_; i++) {
#ifdef USE_VM
        ProtocolState state = bitmap.getEntry<ProtocolState>(block().acceleratorAddr(GetMode()) + i * SubBlockSize_);
#else
        ProtocolState state = ProtocolState(subBlockState_[i]);
#endif
        if (state == lazy::Dirty) {
            if (size == 0) start = i;
            size++;
        } else if (size > 0) {
            ret = Memory::protect(getSubBlockAddr(start), SubBlockSize_ * size, GMAC_PROT_READWRITE);
            if (ret < 0) break;
            size = 0;
        }
    }
    if (size > 0) {
        ret = Memory::protect(getSubBlockAddr(start), SubBlockSize_ * size, GMAC_PROT_READWRITE);
    }
    return ret;
}

inline int
BlockState::protect(GmacProtection prot)
{
    int ret = 0;
#if 1
    ret = Memory::protect(block().addr(), block().size(), prot);
#else
    for (unsigned i = 0; i < subBlockState_.size(); i++) {
        if (subBlockState_[i] == lazy::Dirty) {
            ret = Memory::protect(getSubBlockAddr(i), SubBlockSize_, prot);
            if (ret < 0) break;
        }
    }
#endif
    return ret;
}

inline
void
BlockState::acquired()
{
    //setAll(lazy::Invalid);   
    //state_ = lazy::Invalid;
#ifdef DEBUG
    //::memset(&subBlockFaults_[0], 0, subBlockFaults_.size() * sizeof(long_t));
#endif
}

inline
void
BlockState::released()
{
    //setAll(lazy::ReadOnly);   
    state_ = lazy::ReadOnly;
    reset();
#ifdef DEBUG
    //::memset(&subBlockFaults_[0], 0, subBlockFaults_.size() * sizeof(long_t));
#endif
}

inline
gmacError_t
BlockState::dump(std::ostream &stream, common::Statistic stat)
{
#ifdef DEBUG
    if (stat == common::PAGE_FAULTS_READ) {
        for (unsigned i = 0; i < subBlockFaultsRead_.size(); i++) {
            std::ostringstream oss;
            oss << subBlockFaultsRead_[i] << " ";

            stream << oss.str();
        }
        ::memset(&subBlockFaultsRead_[0], 0, subBlockFaultsRead_.size() * sizeof(long_t));
    } else if (stat == common::PAGE_FAULTS_WRITE) {
        for (unsigned i = 0; i < subBlockFaultsWrite_.size(); i++) {
            std::ostringstream oss;
            oss << subBlockFaultsWrite_[i] << " ";

            stream << oss.str();
        }
        ::memset(&subBlockFaultsWrite_[0], 0, subBlockFaultsWrite_.size() * sizeof(long_t));

    } else if (stat == common::PAGE_TRANSFERS_TO_ACCELERATOR) {
        for (unsigned i = 0; i < transfersToAccelerator_.size(); i++) {
            std::ostringstream oss;
            oss << transfersToAccelerator_[i] << " ";

            stream << oss.str();
        }
        ::memset(&transfersToAccelerator_[0], 0, transfersToAccelerator_.size() * sizeof(long_t));
    } else if (stat == common::PAGE_TRANSFERS_TO_HOST) {
        for (unsigned i = 0; i < transfersToHost_.size(); i++) {
            std::ostringstream oss;
            oss << transfersToHost_[i] << " ";

            stream << oss.str();
        }
        ::memset(&transfersToHost_[0], 0, transfersToHost_.size() * sizeof(long_t));
    }
#endif
    return gmacSuccess;
}

}}}}

#else

namespace __impl {
namespace memory {
namespace protocol {
namespace lazy {

inline
Block &BlockState::block()
{
	return static_cast<Block &>(*this);
}

inline
BlockState::BlockState(ProtocolState init) :
    common::BlockState<lazy::State>(init)
#ifdef DEBUG
    , faultsRead_(0),
    faultsWrite_(0),
    transfersToAccelerator_(0),
    transfersToHost_(0)
#endif
{
}

#if 0
ProtocolState
BlockState::getState(hostptr_t /* addr */) const
{
    return state_;
}
#endif

inline void
BlockState::setState(ProtocolState state, hostptr_t /* addr */)
{
    state_ = state;
}

inline
gmacError_t
BlockState::syncToAccelerator()
{
    TRACE(LOCAL, "Transfer block to accelerator: %p", block().addr());

#ifdef DEBUG
    transfersToAccelerator_++;
#endif
    return block().toAccelerator();
}

inline
gmacError_t
BlockState::syncToHost()
{
    TRACE(LOCAL, "Transfer block to host: %p", block().addr());

#ifdef DEBUG
    transfersToHost_++;
#endif
    return block().toHost();
}

inline
void
BlockState::read(const hostptr_t /*addr*/)
{
#ifdef DEBUG
    faultsRead_++;
#endif
    faultsCacheRead_++;
}

inline
void
BlockState::write(const hostptr_t /*addr*/)
{
#ifdef DEBUG
    faultsWrite_++;
#endif
    faultsCacheWrite_++;
}

inline
bool
BlockState::is(ProtocolState state) const
{
    return state_ == state;
}

inline
int
BlockState::protect(GmacProtection prot)
{
    return Memory::protect(block().addr(), block().size(), prot);
}

inline
int
BlockState::unprotect()
{
    return Memory::protect(block().addr(), block().size(), GMAC_PROT_READWRITE);
}

inline
void
BlockState::acquired()
{
#ifdef DEBUG
    faultsRead_ = 0;
    faultsWrite_ = 0;
#endif

    state_ = lazy::Invalid;
}

inline
void
BlockState::released()
{
#ifdef DEBUG
    faultsRead_ = 0;
    faultsWrite_ = 0;
#endif

    state_ = lazy::ReadOnly;
}

inline
gmacError_t
BlockState::dump(std::ostream &stream, common::Statistic stat)
{
#ifdef DEBUG
    if (stat == common::PAGE_FAULTS_READ) {
        std::ostringstream oss;
        oss << faultsRead_ << " ";
        stream << oss.str();

        faultsRead_ = 0;
    } else if (stat == common::PAGE_FAULTS_WRITE) {
        std::ostringstream oss;
        oss << faultsWrite_ << " ";
        stream << oss.str();

        faultsWrite_ = 0;
    } else if (stat == common::PAGE_TRANSFERS_TO_ACCELERATOR) {
        std::ostringstream oss;
        oss << transfersToAccelerator_ << " ";
        stream << oss.str();

        transfersToAccelerator_ = 0;
    } else if (stat == common::PAGE_TRANSFERS_TO_HOST) {
        std::ostringstream oss;
        oss << transfersToHost_ << " ";
        stream << oss.str();

        transfersToHost_ = 0;
    }
#endif
    return gmacSuccess;
}

}}}}

#endif

#endif /* BLOCKSTATE_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
