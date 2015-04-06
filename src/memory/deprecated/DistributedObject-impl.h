#ifndef GMAC_MEMORY_DISTRIBUTEDOBJECT_INST_H_
#define GMAC_MEMORY_DISTRIBUTEDOBJECT_INST_H_

#include "core/Mode.h"
#include "memory/DistributedBlock.h"

namespace __impl { namespace memory {

template<typename State>
inline void
DistributedObject<State>::modifiedObject()
{
    AcceleratorMap::iterator i;
    for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
        std::list<core::Mode *> modes = i->second;
        std::list<core::Mode *>::iterator j;
        for(j = i->second.begin(); j != i->second.end(); j++) {
            (*j)->modifiedObjects();
        }
    }
}

template<typename State>
DistributedObject<State>::DistributedObject(Protocol &protocol, core::Mode &owner,
                                            hostptr_t hostAddr, size_t size, typename State::ProtocolState init, gmacError_t &err) :
    Object(hostAddr, size)
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

    hostptr_t mark = addr_;
    ptroff_t offset = 0;
    while(size > 0) {
        size_t blockSize = (size > BlockSize_) ? BlockSize_ : size;
        mark += blockSize;
        blocks_.insert(BlockMap::value_type(mark,
                       new DistributedBlock<State>(protocol, addr_ + offset,
                                                   shadow_ + offset, blockSize, init)));
        size -= blockSize;
        offset += ptroff_t(blockSize);
    }
    TRACE(LOCAL, "Creating Distributed Object @ %p : shadow @ %p) ", addr_, shadow_);
}


template<typename State>
DistributedObject<State>::~DistributedObject()
{
    AcceleratorMap::iterator i;
    for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
        std::list<core::Mode *> modes = i->second;
        if (modes.size() > 0) {
            modes.front()->unmap(addr_, size_);
        }
    }
    if (shadow_ != NULL) Memory::unshadow(shadow_, size_);
    if (addr_ != NULL) Memory::unmap(addr_, size_);
    TRACE(LOCAL, "Destroying Distributed Object @ %p", addr_);
}

template<typename State>
inline accptr_t
DistributedObject<State>::acceleratorAddr(core::Mode &current, const hostptr_t addr) const
{
    accptr_t ret = accptr_t(0);
    lockRead();
    BlockMap::const_iterator i = blocks_.upper_bound(addr);
    if(i != blocks_.end()) {
        ret = i->second->acceleratorAddr(current, addr);
    }
    unlock();
    return ret;
}

template<typename State>
inline core::Mode &
DistributedObject<State>::owner(core::Mode &current, const hostptr_t addr) const
{
    lockRead();
    BlockMap::const_iterator i = blocks_.upper_bound(addr);
    ASSERTION(i != blocks_.end());
    core::Mode &ret = i->second->owner(current);
    unlock();
    return ret;
}

template<typename State>
gmacError_t
DistributedObject<State>::addOwner(core::Mode &mode)
{
    accptr_t acceleratorAddr = accptr_t(0);
#ifdef USE_VM
    gmacError_t ret =
                mode.map(acceleratorAddr, addr_, size_, unsigned(SubBlockSize_));
#else
    gmacError_t ret =
                mode.map(acceleratorAddr, addr_, size_);
#endif
    if(ret != gmacSuccess) return ret;

    lockWrite();

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
        DistributedBlock<State> &block = dynamic_cast<DistributedBlock<State> &>(*i->second);
        block.addOwner(mode, acceleratorAddr + offset);
    }
    TRACE(LOCAL, "Add owner %p Object @ %p", &mode, addr_);
    unlock();
    return gmacSuccess;
}

template<typename State>
gmacError_t
DistributedObject<State>::removeOwner(core::Mode &mode)
{
    TRACE(LOCAL, "Remove owner %p Object @ %p", &mode, addr_);

    lockWrite();
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

    BlockMap::iterator j;
    for(j = blocks_.begin(); j != blocks_.end(); j++) {
        DistributedBlock<State> &block = dynamic_cast<DistributedBlock<State> &>(*j->second);
        block.removeOwner(mode);
    }
    //i->first->unmap(i->second, size_);
    //if(acceleratorAddr_.empty()) Map::insertOrphan(*this);

    unlock();
    return gmacSuccess;
}

template<typename State>
inline gmacError_t
DistributedObject<State>::mapToAccelerator()
{
    // TODO Fail
    return gmacSuccess;
}

template<typename State>
inline gmacError_t
DistributedObject<State>::unmapFromAccelerator()
{
    // TODO Fail
    return gmacSuccess;
}

}}

#endif
