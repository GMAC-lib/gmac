#ifndef GMAC_MEMORY_ALLOCATOR_CACHE_IPP_
#define GMAC_MEMORY_ALLOCATOR_CACHE_IPP_

#include "memory/Memory.h"

namespace __impl { namespace memory { namespace allocator {

inline
hostptr_t Arena::key() const
{
    ASSERTION(ptr_ != NULL);
    return ptr_ + memory::BlockSize_;
}

inline
const ObjectList &Arena::objects() const
{
    ASSERTION(ptr_ != NULL);
    return objects_;
}

inline
bool Arena::valid() const
{
    return ptr_ != NULL;
}

inline
bool Arena::full() const
{
    ASSERTION(ptr_ != NULL);
    return objects_.size() == size_;
}

inline
bool Arena::empty() const
{
    ASSERTION(ptr_ != NULL);
    return objects_.empty();
}

inline
hostptr_t Arena::get()
{
    ASSERTION(ptr_ != NULL);
    ASSERTION(objects_.empty() == false);
    hostptr_t ret = objects_.front();
    objects_.pop_front();
    TRACE(LOCAL,"Arena %p has "FMT_SIZE" available objects", this, objects_.size());
    return ret;
}

inline
void Arena::put(hostptr_t obj)
{
    ASSERTION(ptr_ != NULL);
    objects_.push_back(obj);
}

inline
Cache::Cache(Manager &manager, core::Mode &mode, size_t size) :
    gmac::util::Lock("Cache"),
    objectSize(size),
    arenaSize(memory::BlockSize_),
    manager_(manager),
    mode_(mode)
{ }


inline
void Cache::put(hostptr_t obj)
{
    lock();
    ArenaMap::iterator i;
    i = arenas.upper_bound(obj);
    CFATAL(i != arenas.end(), "Address for invalid arena: %p", obj);
    CFATAL(i->second->address() <= obj, "Address for invalid arena: %p", obj);
    i->second->put(obj);
    if(i->second->full()) {
        delete i->second;
        arenas.erase(i);
    }
    unlock();
}

}}}

#endif
