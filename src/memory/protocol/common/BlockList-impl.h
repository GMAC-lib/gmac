#ifndef GMAC_MEMORY_PROTOCOL_BLOCKLIST_IMPL_H_
#define GMAC_MEMORY_PROTOCOL_BLOCKLIST_IMPL_H_

#include <algorithm>

#include "memory/Block.h"
#include "memory/vm/Model.h"

namespace __impl { namespace memory { namespace protocol {


inline BlockList::BlockList() :
#if defined(__APPLE__)
    Lock("BlockList")
#else
    SpinLock("BlockList")
#endif
{}

inline BlockList::~BlockList()
{}

inline bool BlockList::empty() const
{
    lock();
    bool ret = Parent::empty();
    unlock();
    return ret;
}

inline size_t BlockList::size() const
{
    lock();
    size_t ret = Parent::size();
    unlock();
    return ret;
}

inline void BlockList::push(Block &block)
{
    block.incRef();
    lock();
    Parent::push_back(&block);
    unlock();
}

inline Block &BlockList::front()
{
    ASSERTION(Parent::empty() == false);
    lock();
    Block *ret = Parent::front();
    unlock();
    ASSERTION(ret != NULL);
    ret->decRef();
    return *ret;
}

inline void BlockList::remove(Block &block)
{
    lock();
    Parent::remove(&block);
    unlock();
    return;
}

}}}

#endif
