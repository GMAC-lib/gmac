#ifndef GMAC_MEMORY_BLOCKGROUP_INST_H_
#define GMAC_MEMORY_BLOCKGROUP_INST_H_

#include "core/Mode.h"
#include "GenericBlock.h"

namespace __impl { namespace memory {

template<typename State>
inline void BlockGroup<State>::modifiedObject()
{
    AcceleratorMap::iterator i;
    for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
        std::list<core::Mode *> modes = i->second;
        std::list<core::Mode *>::iterator j;
        for(j = i->second.begin(); j != i->second.end(); j++) {
            ObjectMap &map = (*j)->getAddressSpace();
            map.modifiedObjects();
        }
    }
}

static inline gmacError_t
mallocAccelerator(core::Mode &mode, hostptr_t addr, size_t size, accptr_t &acceleratorAddr)
{
    acceleratorAddr = accptr_t(0);
    // Allocate accelerator memory
#ifdef USE_VM
    gmacError_t ret = mode.map(acceleratorAddr, addr, size, unsigned(SubBlockSize_));
    if(ret == gmacSuccess) {
        vm::Bitmap &bitmap = mode.getBitmap();
        bitmap.registerRange(acceleratorAddr, size);
    }
#else
    gmacError_t ret = mode.map(acceleratorAddr, addr, size);
#endif
    return ret;
}

static inline gmacError_t
mallocHost(accptr_t addr, size_t size, hostptr_t &hostAddr)
{
    gmacError_t ret = gmacSuccess;

    hostAddr = NULL;

    // Allocate host memory
    if(hostAddr == NULL) {
        hostAddr = Memory::map(hostptr_t(addr.get()), size, GMAC_PROT_READWRITE);
        if (hostAddr == NULL) {
            return gmacErrorMemoryAllocation;
        }
    }

    return ret;
}

template<typename State>
gmacError_t
BlockGroup<State>::populateBlocks()
{
    // Create memory blocks
    hostptr_t mark = addr_;
    ptroff_t offset = 0;
    size_t size = size_; 
    while(size > 0) {
        size_t blockSize = (size > BlockSize_) ? BlockSize_ : size;
        mark += blockSize;
        blocks_.insert(BlockMap::value_type(mark,
                       new GenericBlock<State>(protocol_, addr_ + offset,
                           shadow_ + offset, blockSize, init_)));
        size -= blockSize;
        offset += ptroff_t(blockSize);
        TRACE(LOCAL, "Creating BlockGroup @ %p : shadow @ %p ("FMT_SIZE" bytes) ", addr_, shadow_, blockSize);
    }
    return gmacSuccess;
}

template<typename State>
gmacError_t
BlockGroup<State>::repopulateBlocks(accptr_t accPtr, core::Mode &mode)
{
    // Repopulate the block-set
    ptroff_t offset = 0;
    for (BlockMap::iterator i = blocks_.begin(); i != blocks_.end(); i++) {
        GenericBlock<State> &oldBlock = *dynamic_cast<GenericBlock<State> *>(i->second);
        GenericBlock<State> *newBlock = new GenericBlock<State>(oldBlock.getProtocol(),
                                                                addr_   + offset,
                                                                shadow_ + offset,
                                                                oldBlock.size(), oldBlock.getState());

        newBlock->addOwner(mode, accPtr + offset);
        i->second = newBlock;

        offset += ptroff_t(oldBlock.size());

        // Decrement reference count
        oldBlock.decRef();
    }

    return gmacSuccess;
}

template<typename State>
BlockGroup<State>::BlockGroup(Protocol &protocol, core::Mode &owner,
                              hostptr_t hostAddr, size_t size, typename State::ProtocolState init, gmacError_t &err) :
    Object(hostAddr, size),
    hasUserMemory_(hostAddr != NULL),
    owners_(0),
    ownerShortcut_(NULL),
    protocol_(protocol),
    init_(init)
{
    shadow_ = NULL;
    err = gmacSuccess;

}


template<typename State>
BlockGroup<State>::~BlockGroup()
{
    AcceleratorMap::iterator i;
    for (i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
        std::list<core::Mode *> modes = i->second;
        if (modes.size() > 0) {
            modes.front()->unmap(addr_, size_);
        }
#ifdef USE_VM
        std::list<core::Mode *>::iterator j;
        for (j = i->second->begin(); j != i->second->end; j++) {
            vm::Bitmap &bitmap = (*j)->getBitmap();
            bitmap.unregisterRange(acceleratorAddr_, size_);
        }
#endif
    }
    if (shadow_ != NULL) Memory::unshadow(shadow_, size_);
    if (addr_ != NULL && hasUserMemory_ == false) Memory::unmap(addr_, size_);
    TRACE(LOCAL, "Destroying BlockGroup @ %p", addr_);
}

template<typename State>
inline accptr_t
BlockGroup<State>::acceleratorAddr(core::Mode &current, const hostptr_t addr) const
{
    accptr_t ret = accptr_t(0);
    lockRead();
    BlockMap::const_iterator i = blocks_.upper_bound(addr);
    if (i != blocks_.end()) {
        ret = i->second->acceleratorAddr(current, addr);
    }
    unlock();
    return ret;
}

template<typename State>
core::Mode &
BlockGroup<State>::owner(core::Mode &current, const hostptr_t addr) const
{
    core::Mode *ret;
    lockRead();
    if (owners_ == 1) {
        ret = ownerShortcut_;
    } else {
        BlockMap::const_iterator i = blocks_.upper_bound(addr);
        ASSERTION(i != blocks_.end());
        ret = &(i->second->owner(current));
    }
    unlock();
    return *ret;
}

template<typename State>
gmacError_t
BlockGroup<State>::addOwner(core::Mode &mode)
{
    TRACE(LOCAL, "Add owner %p Object @ %p", &mode, addr_);
    accptr_t acceleratorAddr = accptr_t(0);

    gmacError_t ret = mallocAccelerator(mode, addr_, size_, acceleratorAddr);
    if (ret != gmacSuccess) return ret;

    lockWrite();

    // Allocate memory (if necessary)
    if (blocks_.size() == 0) {
        if (mode.hasUnifiedAddressing()) {
            ret = mallocHost(acceleratorAddr, size_, addr_);
        } else {
            ret = mallocHost(accptr_t(0), size_, addr_);
        }
        if (ret != gmacSuccess) {
            unlock();
            return ret;
        }

        // Register the mapping for the first allocation
        ret = mode.add_mapping(acceleratorAddr, addr_, size_);
        if (ret != gmacSuccess) {
            unlock();
            return ret;
        }

        // Create a shadow mapping for the host memory
        shadow_ = hostptr_t(Memory::shadow(addr_, size_));
        if (shadow_ == NULL) {
            ret = gmacErrorMemoryAllocation;
            unlock();
            return ret;
        }

        ret = populateBlocks();
        if (ret != gmacSuccess) {
            unlock();
            return ret;
        }
    }
    
    AcceleratorMap::iterator it = acceleratorAddr_.find(acceleratorAddr);
    if (it == acceleratorAddr_.end()) {
        acceleratorAddr_.insert(AcceleratorMap::value_type(acceleratorAddr, std::list<core::Mode *>()));
        AcceleratorMap::iterator it = acceleratorAddr_.find(acceleratorAddr);
        it->second.push_back(&mode);
    } else {
        it->second.push_back(&mode);
    }
    BlockMap::iterator i;
    for(i = blocks_.begin(); i != blocks_.end(); i++) {
        ptroff_t offset = ptroff_t(i->second->addr() - addr_);
        GenericBlock<State> &block = dynamic_cast<GenericBlock<State> &>(*i->second);
        block.addOwner(mode, acceleratorAddr + offset);
    }
    if (owners_ == 0) {
        ownerShortcut_ = &mode;
    } else {
        ownerShortcut_ = NULL;
    }
    owners_++;
    unlock();
    return gmacSuccess;
}

// TODO: move checks to DBC
template<typename State>
gmacError_t
BlockGroup<State>::removeOwner(core::Mode &mode)
{
    lockWrite();

    TRACE(LOCAL, "Remove owner %p Object @ %p: %u -> %u", &mode, addr_, owners_, owners_ - 1);

    ASSERTION(owners_ > 0);

    if (owners_ == 1) {
        ASSERTION(acceleratorAddr_.size() == 1);

        if (!hasUserMemory_) {
            // TODO: Put myself in the orphan map
            ownerShortcut_->makeOrphan(*this);
            TRACE(LOCAL, "BlockGroup @ %p is going orphan", addr_);
        } else {
            TRACE(LOCAL, "BlockGroup @ %p is NOT going orphan", addr_);
        }

        gmacError_t ret = coherenceOp(&Protocol::deleteBlock);
        ASSERTION(ret == gmacSuccess);
        ret = coherenceOp(&Protocol::unmapFromAccelerator);
        ASSERTION(ret == gmacSuccess);
        ownerShortcut_->unmap(addr_, size_);

        acceleratorAddr_.clear();

        // Clean-up
        BlockMap::iterator i;
        for(i = blocks_.begin(); i != blocks_.end(); i++) {
            i->second->decRef();
        }
        blocks_.clear();

        ownerShortcut_ = NULL;
    } else {
        AcceleratorMap::iterator i;
        bool ownerFound = false;
        for (i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
            std::list<core::Mode *> &list = i->second;
            std::list<core::Mode *>::iterator j = std::find(list.begin(), list.end(), &mode);
            if (j != list.end()) {
                list.erase(j);
                if (list.size() == 0) {
                    acceleratorAddr_.erase(i);
                    mode.unmap(addr_, size_);
                }
                ownerFound = true;
                break;
            }
        }
        ASSERTION(ownerFound == true);

        for (BlockMap::iterator j = blocks_.begin(); j != blocks_.end(); j++) {
            GenericBlock<State> &block = dynamic_cast<GenericBlock<State> &>(*j->second);
            block.removeOwner(mode);
        }

        if (owners_ == 2) {
            ASSERTION(acceleratorAddr_.size() == 1);
            i = acceleratorAddr_.begin();
            std::list<core::Mode *> &list = i->second;
            ASSERTION(list.size() == 1);
            ownerShortcut_ = list.front();
        }
    }

    owners_--;
    unlock();
    return gmacSuccess;
}

// TODO Receive a mode
template<typename State>
inline gmacError_t
BlockGroup<State>::mapToAccelerator()
{
    gmacError_t ret;

    lockWrite();

    if (owners_ == 1) {
        // Allocate accelerator memory in the new mode
        accptr_t newAcceleratorAddr(0);

        ret = mallocAccelerator(*ownerShortcut_, addr_, size_, newAcceleratorAddr);

        if (ret == gmacSuccess) {
            ASSERTION(acceleratorAddr_.size() == 1);
            acceleratorAddr_.clear();
            acceleratorAddr_.insert(AcceleratorMap::value_type(newAcceleratorAddr, std::list<core::Mode *>()));
            AcceleratorMap::iterator it = acceleratorAddr_.find(newAcceleratorAddr);
            it->second.push_back(ownerShortcut_);
            
            // Recreate accelerator blocks
            repopulateBlocks(newAcceleratorAddr, *ownerShortcut_);
            // Add blocks to the coherence domain
            ret = coherenceOp(&Protocol::mapToAccelerator);
        }
    } else {
        // Not supported for now
        return gmacErrorFeatureNotSupported;
    }

    unlock();
    return ret;
}

// TODO Receive a mode
template<typename State>
inline gmacError_t
BlockGroup<State>::unmapFromAccelerator()
{
    gmacError_t ret;

    lockWrite();
    // Not supported for now
    if (owners_ == 1) {
        // Remove blocks from the coherence domain
        ret = coherenceOp(&Protocol::unmapFromAccelerator);

        // Free accelerator memory
        if (ret == gmacSuccess) {
            ret = ownerShortcut_->unmap(addr_, size_);
            ASSERTION(ret == gmacSuccess, "Error unmapping object from accelerator");
        }
    } else {
        ret = gmacErrorFeatureNotSupported;
    }
    unlock();
    return ret;
}

}}

#endif
