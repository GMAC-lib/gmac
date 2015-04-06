#ifndef GMAC_MEMORY_SHAREDOBJECT_IMPL_H_
#define GMAC_MEMORY_SHAREDOBJECT_IMPL_H_

#include "core/Mode.h"

#include "SharedBlock.h"

namespace __impl { namespace memory {

template<typename State>
void SharedObject<State>::modifiedObject()
{
    if(owner_ != NULL) owner_->modifiedObjects();
}

static gmacError_t
allocAcceleratorMemory(core::Mode &mode, hostptr_t addr, size_t size, accptr_t &acceleratorAddr)
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

template<typename State>
gmacError_t
SharedObject<State>::repopulateBlocks(accptr_t accPtr, core::Mode &mode)
{
    // Repopulate the block-set
    ptroff_t offset = 0;
    for(BlockMap::iterator i = blocks_.begin(); i != blocks_.end(); i++) {
        SharedBlock<State> &oldBlock = *dynamic_cast<SharedBlock<State> *>(i->second);
        SharedBlock<State> *newBlock = new SharedBlock<State>(oldBlock.getProtocol(), mode,
                                                      addr_   + offset,
                                                      shadow_ + offset,
                                                      accPtr  + offset,
                                                      oldBlock.size(), oldBlock.getState());

        i->second = newBlock;

        offset += ptroff_t(oldBlock.size());

        // Decrement reference count
        oldBlock.decRef();
    }

    return gmacSuccess;
}

template<typename State>
SharedObject<State>::SharedObject(Protocol &protocol, core::Mode &owner,
                                  hostptr_t hostAddr, size_t size, typename State::ProtocolState init, gmacError_t &err) :
    Object(hostAddr, size),
    acceleratorAddr_(0),
    owner_(&owner)
{
    shadow_ = NULL;
    err = gmacSuccess;

    // Allocate memory (if necessary)
    if(hostAddr == NULL) {
        addr_ = Memory::map(NULL, size, GMAC_PROT_READWRITE);
        if (addr_ == NULL) {
            err = gmacErrorMemoryAllocation;
            return;
        }
    }

    // Create a shadow mapping for the host memory
    shadow_ = hostptr_t(Memory::shadow(addr_, size_));
    if (shadow_ == NULL) {
        err = gmacErrorMemoryAllocation;
        return;
    }

    // Allocate accelerator memory
    err = allocAcceleratorMemory(owner, addr_, size, acceleratorAddr_);
    if (err != gmacSuccess) return;

        // Populate the block-set
    hostptr_t mark = addr_;
    ptroff_t offset = 0;
    while(size > 0) {
        size_t blockSize = (size > BlockSize_) ? BlockSize_ : size;
        mark += blockSize;
        blocks_.insert(BlockMap::value_type(mark,
                       new SharedBlock<State>(protocol, owner, addr_ + ptroff_t(offset),
                                              shadow_ + offset, acceleratorAddr_ + offset, blockSize, init)));
        size -= blockSize;
        offset += ptroff_t(blockSize);
    }
    TRACE(LOCAL, "Creating Shared Object @ %p : shadow @ %p : accelerator @ %p) ", addr_, shadow_, (void *) acceleratorAddr_);
}


template<typename State>
SharedObject<State>::~SharedObject()
{
#ifdef USE_VM
    vm::Bitmap &bitmap = owner_->getBitmap();
    bitmap.unregisterRange(acceleratorAddr_, size_);
#endif

    // If the object creation failed, this address will be NULL
    if (acceleratorAddr_ != 0) owner_->unmap(addr_, size_);
    if (shadow_ != NULL) Memory::unshadow(shadow_, size_);
    if (addr_ != NULL) Memory::unmap(addr_, size_);
    TRACE(LOCAL, "Destroying Shared Object @ %p", addr_);
}

template<typename State>
inline accptr_t
SharedObject<State>::acceleratorAddr(core::Mode &current, const hostptr_t addr) const
{
    printf("HOLA\n");
    accptr_t ret = accptr_t(0);
    lockRead();
    if(acceleratorAddr_ != 0) {
        ptroff_t offset = ptroff_t(addr - addr_);
        ret = acceleratorAddr_ + offset;
    }
    unlock();
    return ret;
}

template<typename State>
inline core::Mode &
SharedObject<State>::owner(core::Mode &current, const hostptr_t addr) const
{
    lockRead();
    core::Mode &ret = *owner_;
    unlock();
    return ret;
}

template<typename State>
inline gmacError_t
SharedObject<State>::addOwner(core::Mode &owner)
{
    TRACE(LOCAL, "Add owner %p Object @ %p", &owner, addr_);

    return gmacErrorUnknown; // This kind of objects only accepts one owner
}

template<typename State>
inline gmacError_t
SharedObject<State>::removeOwner(core::Mode &owner)
{
    TRACE(LOCAL, "Remove owner %p Object @ %p", &owner, addr_);

    lockWrite();
    if(owner_ == &owner) {
        // Put myself in the orphan map
        owner.makeOrphan(*this);

        TRACE(LOCAL, "Shared Object @ %p is going orphan", addr_);
        if(acceleratorAddr_ != 0) {
            gmacError_t ret = coherenceOp(&Protocol::deleteBlock);
            ASSERTION(ret == gmacSuccess);
            ret = coherenceOp(&Protocol::unmapFromAccelerator);
            ASSERTION(ret == gmacSuccess);
            owner_->unmap(addr_, size_);
        }
        // Clean-up
        BlockMap::iterator i;
        for(i = blocks_.begin(); i != blocks_.end(); i++) {
            i->second->decRef();
        }
        blocks_.clear();

        acceleratorAddr_ = accptr_t(0);
        owner_ = NULL;
    }
    unlock();
    return gmacSuccess;
}

template<typename State>
inline gmacError_t
SharedObject<State>::mapToAccelerator()
{
    gmacError_t ret;

    lockWrite();

    // Allocate accelerator memory in the new mode
    accptr_t newAcceleratorAddr(0);

    ret = allocAcceleratorMemory(*owner_, addr_, size_, newAcceleratorAddr);

    if (ret == gmacSuccess) {
        acceleratorAddr_ = newAcceleratorAddr;
        // Recreate accelerator blocks
        repopulateBlocks(acceleratorAddr_, *owner_);
        // Add blocks to the coherency domain
        ret = coherenceOp(&Protocol::mapToAccelerator);
    }

    unlock();
    return ret;
}

template<typename State>
inline gmacError_t
SharedObject<State>::unmapFromAccelerator()
{
    lockWrite();
    // Remove blocks from the coherency domain
    gmacError_t ret = coherenceOp(&Protocol::unmapFromAccelerator);
    // Free accelerator memory
    if (ret == gmacSuccess)
        CFATAL(owner_->unmap(addr_, size_) == gmacSuccess, "Error unmapping object from accelerator");
    unlock();
    return ret;
}

}}

#endif
