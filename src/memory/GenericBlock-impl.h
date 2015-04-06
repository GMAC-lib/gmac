#ifndef GMAC_MEMORY_GENERICBLOCK_INST_H_
#define GMAC_MEMORY_GENERICBLOCK_INST_H_

#include <algorithm>

#include "core/Mode.h"
#include "core/IOBuffer.h"

namespace __impl { namespace memory {

template<typename State>
inline
GenericBlock<State>::GenericBlock(Protocol &protocol, hostptr_t hostAddr,
                                  hostptr_t shadowAddr, size_t size, typename State::ProtocolState init) :
    StateBlock<State>(protocol, hostAddr, shadowAddr, size, init),
    ownerShortcut_(NULL)
{
}

template<typename State>
inline
GenericBlock<State>::~GenericBlock()
{
	/* Decrement the usage count for the owners */
	ModeMap::iterator i;
	for(i = owners_.begin(); i != owners_.end(); i++) {
		i->first->decRef();
	}
}

template<typename State>
void
GenericBlock<State>::addOwner(core::Mode &mode, accptr_t addr)
{
    StateBlock<State>::lock();

    AcceleratorAddrMap::iterator it = acceleratorAddr_.find(addr);

    TRACE(LOCAL, "Adding owner for address for %u:%p @ Context %p", addr.pasId_, addr.get(), &mode);
    if (it == acceleratorAddr_.end()) {
        TRACE(LOCAL, "Adding new address for %u:%p @ Context %p", addr.pasId_, addr.get(), &mode);
        acceleratorAddr_.insert(AcceleratorAddrMap::value_type(addr, std::list<core::Mode *>()));
        AcceleratorAddrMap::iterator it = acceleratorAddr_.find(addr);
        it->second.push_back(&mode);

        if(StateBlock<State>::protocol_.needUpdate(*this) == true &&
           acceleratorAddr_.size() > 1) {
            gmacError_t ret = mode.copyToAccelerator(addr, StateBlock<State>::shadow_, StateBlock<State>::size_);
            ASSERTION(ret == gmacSuccess);
        }
    } else {
        it->second.push_back(&mode);
    }

    ASSERTION(owners_.find(&mode) == owners_.end());
    owners_.insert(ModeMap::value_type(&mode, addr));
    mode.incRef();

    if (owners_.size() == 1) {
        ownerShortcut_ = &mode;
    }

    StateBlock<State>::unlock();
}
// TODO: return error!
template<typename State>
void
GenericBlock<State>::removeOwner(core::Mode &mode)
{
    StateBlock<State>::lock();

    AcceleratorAddrMap::iterator a;
    for (a = acceleratorAddr_.begin(); a != acceleratorAddr_.end(); a++) {
        std::list<core::Mode *> &list = a->second;
        std::list<core::Mode *>::iterator j = std::find(list.begin(), list.end(), &mode);
        if (j != list.end()) {
            list.erase(j);
            if (list.size() == 0) acceleratorAddr_.erase(a);
            goto done_addr;
        }
    }
    FATAL("Mode NOT found!");
done_addr:

    ModeMap::iterator m;
    m = owners_.find(&mode);
    ASSERTION(m != owners_.end());
    owners_.erase(m);
    m->first->decRef();

    StateBlock<State>::unlock();
}

template<typename State>
inline core::Mode &
GenericBlock<State>::owner(core::Mode &current) const
{
    core::Mode *ret;
    ASSERTION(owners_.size() > 0);

    if (owners_.size() == 1) {
        ret = ownerShortcut_;
    } else {
        ModeMap::const_iterator m;
        m = owners_.find(&current);
        if (m == owners_.end()) {
            ret = owners_.begin()->first;
        } else {
            ret = m->first;
        }
    }
    return *ret;
}

template<typename State>
inline accptr_t
GenericBlock<State>::acceleratorAddr(core::Mode &current, const hostptr_t addr) const
{
    accptr_t ret = accptr_t(0);

    //StateBlock<State>::lock();

    ModeMap::const_iterator m;
    if (owners_.size() == 1) {
        m = owners_.begin();
        ret = m->second + (addr - this->addr_);
    } else {
        m = owners_.find(&current);
        if (m != owners_.end()) {
            ret = m->second + (addr - this->addr_);
        }
    }

    //StateBlock<State>::unlock();
    return ret;
}

template<typename State>
inline accptr_t
GenericBlock<State>::acceleratorAddr(core::Mode &current) const
{
    return acceleratorAddr(current, this->addr_);
}

template<typename State>
inline gmacError_t
GenericBlock<State>::toHost(unsigned blockOff, size_t count)
{
    gmacError_t ret = gmacSuccess;

    // Fast path
    if (owners_.size() == 1) {
        ModeMap::const_iterator m;
        m = owners_.begin();
        ret = ownerShortcut_->copyToHost(this->shadow_ + blockOff, m->second + blockOff, count);
    } else { // TODO Implement this path
        ret = gmacSuccess;
    }

    return ret;
}

template<typename State>
inline gmacError_t
GenericBlock<State>::toAccelerator(unsigned blockOff, size_t count)
{
    gmacError_t ret = gmacSuccess;

    // Fast path
    if (owners_.size() == 1) {
        ModeMap::const_iterator m;
        m = owners_.begin();
        ret = ownerShortcut_->copyToAccelerator(m->second + blockOff, StateBlock<State>::shadow_ + blockOff, count);
    } else {
        AcceleratorAddrMap::const_iterator a;
        for(a = acceleratorAddr_.begin(); a != acceleratorAddr_.end(); a++) {
            const std::list<core::Mode *> &list = a->second;
            ASSERTION(list.size() > 0);
            core::Mode *mode = list.front();
            ret = mode->copyToAccelerator(a->first + blockOff, StateBlock<State>::shadow_ + blockOff, count);
            if(ret != gmacSuccess) break;
        }
    }
    return ret;
}

template<typename State>
inline gmacError_t
GenericBlock<State>::copyFromBuffer(size_t blockOff, core::IOBuffer &buffer,
                                    size_t bufferOff, size_t size, typename StateBlock<State>::Destination dst) const
{
    gmacError_t ret = gmacSuccess;
    trace::EnterCurrentFunction();
    switch (dst) {
    case StateBlock<State>::HOST:
        ::memcpy(StateBlock<State>::shadow_ + blockOff, buffer.addr() + bufferOff, size);
        break;

    case StateBlock<State>::ACCELERATOR:
        if (owners_.size() == 1) { // Fast path
            ModeMap::const_iterator m;
            m = owners_.begin();
            ret = ownerShortcut_->bufferToAccelerator(m->second + ptroff_t(blockOff), buffer, size, bufferOff);
        } else {
            AcceleratorAddrMap::const_iterator i;
            for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
                const std::list<core::Mode *> &list = i->second;
                ASSERTION(list.size() > 0);
                core::Mode *mode = list.front();
                ret = mode->bufferToAccelerator(i->first + ptroff_t(blockOff), buffer, size, bufferOff);
                if (ret != gmacSuccess) break;
            }
        }
        break;
    }

    trace::ExitCurrentFunction();
    return ret;
}

template<typename State>
inline gmacError_t
GenericBlock<State>::copyToBuffer(core::IOBuffer &buffer, size_t bufferOff,
                                  size_t blockOff, size_t size, typename StateBlock<State>::Source src) const
{
    gmacError_t ret = gmacSuccess;

    trace::EnterCurrentFunction();
    switch (src) {
    case StateBlock<State>::HOST:
        ::memcpy(buffer.addr() + bufferOff, StateBlock<State>::shadow_ + blockOff, size);
        break;
    case StateBlock<State>::ACCELERATOR:
        if (owners_.size() == 1) { // Fast path
            ModeMap::const_iterator m;
            m = owners_.begin();
            ret = ownerShortcut_->acceleratorToBuffer(buffer, m->second + ptroff_t(blockOff), size, bufferOff);
        } else {
            ret = gmacErrorFeatureNotSupported;
        }
        break;
    }

    trace::ExitCurrentFunction();
    return ret;
}

template<typename State>
gmacError_t
GenericBlock<State>::copyFromBlock(size_t dstOff, StateBlock<State> &srcBlock,
                                   size_t srcOff, size_t size,
                                   typename StateBlock<State>::Destination dst,
                                   typename StateBlock<State>::Source src) const
{
    gmacError_t ret = gmacSuccess;
    if (dst == StateBlock<State>::HOST &&
        src == StateBlock<State>::HOST) {
        ::memcpy(this->shadow_ + dstOff, srcBlock.getShadow() + srcOff, size);
    } else if (src == StateBlock<State>::ACCELERATOR &&
               dst == StateBlock<State>::ACCELERATOR) {
        AcceleratorAddrMap::const_iterator i;
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
        AcceleratorAddrMap::const_iterator i;
        for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
            const std::list<core::Mode *> &list = i->second;
            ASSERTION(list.size() > 0);
            core::Mode *mode = list.front();
            ret = mode->copyToAccelerator(i->first + dstOff, srcBlock.getShadow() + srcOff, size);
            if(ret != gmacSuccess) return ret;
        }
    } else if (src == StateBlock<State>::ACCELERATOR &&
               dst == StateBlock<State>::HOST) {
        if (owners_.size() == 1) { // Fast path
            ModeMap::const_iterator m;
            m = owners_.begin();
            ret = srcBlock.owner(*ownerShortcut_).copyToHost(this->shadow_ + dstOff, srcBlock.acceleratorAddr(*ownerShortcut_) + srcOff, size);
        } else {
            ret = gmacErrorFeatureNotSupported;
        }
    }

    return ret;

}

template<typename State>
gmacError_t
GenericBlock<State>::memset(int v, size_t size, size_t blockOffset, typename StateBlock<State>::Destination dst) const
{
    gmacError_t ret = gmacSuccess;
    if (dst == StateBlock<State>::HOST) {
        ::memset(StateBlock<State>::shadow_ + blockOffset, v, size);
    } else  {
        if (owners_.size() == 1) { // Fast path
            ModeMap::const_iterator m;
            m = owners_.begin();
            ret = ownerShortcut_->memset(m->second + ptroff_t(blockOffset), v, size);
        } else {
            AcceleratorAddrMap::const_iterator i;
            for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
                const std::list<core::Mode *> &list = i->second;
                ASSERTION(list.size() > 0);
                core::Mode *mode = list.front();
                ret = mode->memset(i->first + ptroff_t(blockOffset), v, size);
                if(ret != gmacSuccess) break;
            }
        }
    }
    return ret;
}

}}

#endif
