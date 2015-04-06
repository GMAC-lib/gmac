#ifndef GMAC_MEMORY_OBJECT_IMPL_H_
#define GMAC_MEMORY_OBJECT_IMPL_H_

#include <fstream>
#include <sstream>

#include "Protocol.h"
#include "Block.h"

#include "util/Logger.h"

namespace __impl { namespace memory {

inline Object::Object(hostptr_t addr, size_t size) :
    gmac::util::RWLock("Object"),
    util::Reference("Object"),
    addr_(addr),
    size_(size),
    released_(false)
{
#ifdef DEBUG
    id_ = AtomicInc(Object::Id_);
#endif
}

#ifdef DEBUG
inline unsigned
Object::getId() const
{
    return id_;
}

inline unsigned
Object::getDumps(protocol::common::Statistic stat)
{
    if (dumps_.find(stat) == dumps_.end()) dumps_[stat] = 0;
    return dumps_[stat];
}
#endif

inline hostptr_t
Object::addr() const
{
    // No need for lock -- addr_ is never modified
    return addr_;
}

inline hostptr_t
Object::end() const
{
    // No need for lock -- addr_ and size_ are never modified
    return addr_ + size_;
}

inline ssize_t
Object::blockBase(size_t offset) const
{
    return -1 * (offset % blockSize());
}

inline size_t
Object::blockEnd(size_t offset) const
{
    if (offset + blockBase(offset) + blockSize() > size_)
        return size_ - offset;
    else
        return size_t(ssize_t(blockSize()) + blockBase(offset));
}

inline size_t
Object::blockSize() const
{
    return BlockSize_;
}

inline size_t
Object::size() const
{
    // No need for lock -- size_ is never modified
    return size_;
}

template <typename T>
gmacError_t Object::coherenceOp(gmacError_t (Protocol::*op)(Block &, T &), T &param)
{
    gmacError_t ret = gmacSuccess;
    BlockMap::const_iterator i;
    for(i = blocks_.begin(); i != blocks_.end(); i++) {
        ret = i->second->coherenceOp(op, param);
        if(ret != gmacSuccess) break;
    }
    return ret;
}

inline gmacError_t
Object::acquire(GmacProtection &prot)
{
    lockWrite();
    gmacError_t ret = gmacSuccess;
    TRACE(LOCAL, "Acquiring object %p?", addr_);
    if (released_ == true) {
        TRACE(LOCAL, "Acquiring object %p", addr_);
        ret = coherenceOp<GmacProtection>(&Protocol::acquire, prot);
    }
    released_ = false;
    unlock();
    return ret;
}

inline gmacError_t
Object::release()
{
    lockWrite();
    released_ = true;
    unlock();
    return gmacSuccess;
}

inline gmacError_t
Object::releaseBlocks()
{
    lockWrite();
    gmacError_t ret = gmacSuccess;

    TRACE(LOCAL, "Releasing object %p?", addr_);
    if (released_ == false) {
        TRACE(LOCAL, "Releasing object %p", addr_);
        ret = coherenceOp(&Protocol::release);
    }

    released_ = true;
    unlock();
    return ret;
}

#ifdef USE_VM
inline gmacError_t
Object::acquireWithBitmap()
{
    lockRead();
    gmacError_t ret = coherenceOp(&Protocol::acquireWithBitmap);
    unlock();
    return ret;
}
#endif

template <typename P1, typename P2>
gmacError_t
Object::forEachBlock(gmacError_t (Block::*f)(P1 &, P2), P1 &p1, P2 p2)
{
    lockRead();
    gmacError_t ret = gmacSuccess;
    BlockMap::iterator i;
    for(i = blocks_.begin(); i != blocks_.end(); i++) {
        ret = (i->second->*f)(p1, p2);
    }
    unlock();
    return ret;
}

inline gmacError_t Object::toHost()
{
    lockRead();
    gmacError_t ret= coherenceOp(&Protocol::toHost);
    unlock();
    return ret;
}

inline gmacError_t Object::toAccelerator()
{
    lockRead();
    gmacError_t ret = coherenceOp(&Protocol::release);
    unlock();
    return ret;
}

inline gmacError_t
Object::copyToBuffer(core::IOBuffer &buffer, size_t size,
                     size_t bufferOffset, size_t objectOffset)
{
    trace::EnterCurrentFunction();
    lockRead();
    gmacError_t ret = memoryOp(&Protocol::copyToBuffer, buffer, size,
                               bufferOffset, objectOffset);
    unlock();
    trace::ExitCurrentFunction();
    return ret;
}

inline gmacError_t Object::copyFromBuffer(core::IOBuffer &buffer, size_t size,
                                          size_t bufferOffset, size_t objectOffset)
{
    trace::EnterCurrentFunction();
    lockRead();
    gmacError_t ret = memoryOp(&Protocol::copyFromBuffer, buffer, size,
                               bufferOffset, objectOffset);
    unlock();
    trace::ExitCurrentFunction();
    return ret;
}

#if 0
inline gmacError_t Object::copyObjectToObject(Object &dst, size_t dstOff,
                                              Object &src, size_t srcOff, size_t count)
{
    dst.lockWrite();
    src.lockWrite();
        gmacError_t ret = memoryOp(&Protocol::copyFromBuffer, buffer, size,
        bufferOffset, objectOffset);
    dst.unlock();
    src.unlock();
    return ret;
}
#endif

}}

#endif
