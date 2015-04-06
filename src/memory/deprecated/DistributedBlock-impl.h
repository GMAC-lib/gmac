#ifndef GMAC_MEMORY_DISTRIBUTEDBLOCK_INST_H_
#define GMAC_MEMORY_DISTRIBUTEDBLOCK_INST_H_

#include <algorithm>

#include "core/Mode.h"
#include "core/IOBuffer.h"

namespace __impl { namespace memory {

template<typename State>
inline
DistributedBlock<State>::DistributedBlock(Protocol &protocol, hostptr_t hostAddr,
                                          hostptr_t shadowAddr, size_t size, typename State::ProtocolState init) :
    StateBlock<State>(protocol, hostAddr, shadowAddr, size, init)
{
}

template<typename State>
inline
DistributedBlock<State>::~DistributedBlock()
{}

template<typename State>
void
DistributedBlock<State>::addOwner(core::Mode &mode, accptr_t addr)
{
    StateBlock<State>::lock();

    AcceleratorMap::iterator it = acceleratorAddr_.find(addr);

    TRACE(LOCAL, "Adding owner for address for %u:%p @ Context %p", addr.pasId_, addr.get(), &mode);
    if (it == acceleratorAddr_.end()) {
        TRACE(LOCAL, "Adding new address for %u:%p @ Context %p", addr.pasId_, addr.get(), &mode);
        acceleratorAddr_.insert(AcceleratorMap::value_type(addr, std::list<core::Mode *>()));
        AcceleratorMap::iterator it = acceleratorAddr_.find(addr);
        it->second.push_back(&mode);

        if(StateBlock<State>::protocol_.needUpdate(*this) == true &&
           acceleratorAddr_.size() > 1) {
            gmacError_t ret = mode.copyToAccelerator(addr, StateBlock<State>::shadow_, StateBlock<State>::size_);
            ASSERTION(ret == gmacSuccess);
        }
    } else {
        it->second.push_back(&mode);
    }

    StateBlock<State>::unlock();
}

template<typename State>
void
DistributedBlock<State>::removeOwner(core::Mode &mode)
{
    StateBlock<State>::lock();

    AcceleratorMap::iterator i;
    for (i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
        std::list<core::Mode *> &list = i->second;
        std::list<core::Mode *>::iterator j = std::find(list.begin(), list.end(), &mode);
        if (j != list.end()) {
            list.erase(j);
            if (list.size() == 0) acceleratorAddr_.erase(i);
            break;
        }
    }

    StateBlock<State>::unlock();
}

template<typename State>
inline core::Mode &
DistributedBlock<State>::owner(core::Mode &current) const
{
    return current;
}

template<typename State>
inline accptr_t
DistributedBlock<State>::acceleratorAddr(core::Mode &current, const hostptr_t addr) const
{
    accptr_t ret = accptr_t(0);

    StateBlock<State>::lock();
    AcceleratorMap::const_iterator i;
    TRACE(LOCAL, "Accelerator address for %p @ Context %p", addr, &current);
    for (i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
        const std::list<core::Mode *> &list = i->second;
        std::list<core::Mode *>::const_iterator it;
        it = std::find(list.begin(), list.end(), &current);

        if (it != list.end()) {
            ret = i->first + int(addr - StateBlock<State>::addr_);
            break;
        }

    }
    ASSERTION(i != acceleratorAddr_.end());
    StateBlock<State>::unlock();
    return ret;
}

template<typename State>
inline accptr_t
DistributedBlock<State>::acceleratorAddr(core::Mode &current) const
{
    accptr_t ret = accptr_t(0);

    StateBlock<State>::lock();

    AcceleratorMap::const_iterator i;
    for (i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
        const std::list<core::Mode *> &list = i->second;
        if (std::find(list.begin(), list.end(), &current) != list.end()) {
            ret = i->first;
            break;
        }
    }

    StateBlock<State>::unlock();
    return ret;
}

template<typename State>
inline gmacError_t
DistributedBlock<State>::toHost(unsigned /*blockOff*/, size_t /*count*/)
{
        return gmacSuccess;
}

template<typename State>
inline gmacError_t
DistributedBlock<State>::toAccelerator(unsigned blockOff, size_t count)
{
    gmacError_t ret = gmacSuccess;
    AcceleratorMap::const_iterator i;
    for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
        const std::list<core::Mode *> &list = i->second;
        ASSERTION(list.size() > 0);
        core::Mode *mode = list.front();
        ret = mode->copyToAccelerator(i->first + blockOff, StateBlock<State>::shadow_ + blockOff, count);
        if(ret != gmacSuccess) break;
    }
    return ret;
}

template<typename State>
inline gmacError_t
DistributedBlock<State>::copyFromBuffer(size_t blockOff, core::IOBuffer &buffer,
                                        size_t bufferOff, size_t size, typename StateBlock<State>::Destination dst) const
{
    gmacError_t ret = gmacSuccess;

    switch (dst) {
    case StateBlock<State>::HOST:
        ::memcpy(StateBlock<State>::shadow_ + blockOff, buffer.addr() + bufferOff, size);
        break;

    case StateBlock<State>::ACCELERATOR:
        AcceleratorMap::const_iterator i;
        for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
            const std::list<core::Mode *> &list = i->second;
            ASSERTION(list.size() > 0);
            core::Mode *mode = list.front();
            ret = mode->bufferToAccelerator(i->first + ptroff_t(blockOff), buffer, size, bufferOff);
            if(ret != gmacSuccess) return ret;
        }
        break;
    }

    return ret;
}

template<typename State>
inline gmacError_t DistributedBlock<State>::copyToBuffer(core::IOBuffer &buffer, size_t bufferOff,
    size_t blockOff, size_t size, typename StateBlock<State>::Source /*src*/) const
{
        ::memcpy(buffer.addr() + bufferOff, StateBlock<State>::shadow_ + blockOff, size);
        return gmacSuccess;
}

template<typename State>
gmacError_t
DistributedBlock<State>::copyFromBlock(size_t dstOff, StateBlock<State> &srcBlock,
                                       size_t srcOff, size_t size,
                                       typename StateBlock<State>::Destination dst,
                                       typename StateBlock<State>::Source src) const
{
    gmacError_t ret = gmacSuccess;
    if (dst == StateBlock<State>::HOST) {
        ::memcpy(this->shadow_ + dstOff, srcBlock.getShadow() + srcOff, size);
    } else if (src == StateBlock<State>::ACCELERATOR &&
               dst == StateBlock<State>::ACCELERATOR) {
        AcceleratorMap::const_iterator i;
        for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
            const std::list<core::Mode *> &list = i->second;
            ASSERTION(list.size() > 0);
            core::Mode &mode = *list.front();
            accptr_t srcPtr = srcBlock.acceleratorAddr(mode) + srcOff;
            if (i->first.pasId_ != srcPtr.pasId_) {
                ret = mode.copyToAccelerator(i->first + dstOff, srcBlock.getShadow() + srcOff, size);
            } else {
                ret = mode.copyAccelerator(i->first + dstOff, srcPtr, size);
            }
            if(ret != gmacSuccess) return ret;
        }
    } else if (src == StateBlock<State>::HOST &&
               dst == StateBlock<State>::ACCELERATOR) {
        AcceleratorMap::const_iterator i;
        for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
            const std::list<core::Mode *> &list = i->second;
            ASSERTION(list.size() > 0);
            core::Mode *mode = list.front();
            ret = mode->copyToAccelerator(i->first + dstOff, srcBlock.getShadow() + srcOff, size);
            if(ret != gmacSuccess) return ret;
        }
    }

    return ret;

}

template<typename State>
gmacError_t
DistributedBlock<State>::memset(int v, size_t size, size_t blockOffset, typename StateBlock<State>::Destination dst) const
{
    gmacError_t ret = gmacSuccess;
    if (dst == StateBlock<State>::HOST) {
        ::memset(StateBlock<State>::shadow_ + blockOffset, v, size);
    } else  {
        AcceleratorMap::const_iterator i;
        for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
        const std::list<core::Mode *> &list = i->second;
        ASSERTION(list.size() > 0);
        core::Mode *mode = list.front();
        ret = mode->memset(i->first + ptroff_t(blockOffset), v, size);
                if(ret != gmacSuccess) break;
        }
    }
    return ret;
}

}}

#endif
