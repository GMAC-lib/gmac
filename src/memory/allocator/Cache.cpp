#include "Cache.h"

#include "core/Mode.h"
#include "memory/Manager.h"

#include "util/Parameter.h"
#include "util/Private.h"

namespace __impl { namespace memory { namespace allocator {

Arena::Arena(Manager &manager, core::Mode &mode, size_t objSize) :
    ptr_(NULL),
    size_(0),
    manager_(manager),
    mode_(mode)
{
    mode_.incRef();
    gmacError_t ret = manager_.alloc(mode_, &ptr_, memory::BlockSize_);
    if(ret != gmacSuccess) { ptr_ = NULL; return; }
    for(size_t s = 0; s < memory::BlockSize_; s += objSize, size_++) {
        TRACE(LOCAL,"Arena %p pushes %p ("FMT_SIZE" bytes)", this, (void *)(ptr_ + s), objSize);
        objects_.push_back(ptr_ + s);
    }
}

Arena::~Arena()
{
    CFATAL(objects_.size() == size_, "Destroying non-full Arena");
    objects_.clear();
    if(ptr_ != NULL) {
        gmacError_t ret = manager_.free(mode_, ptr_);
        ASSERTION(ret == gmacSuccess);
    }
    mode_.decRef();
}

Cache::~Cache()
{
    ArenaMap::iterator i;
    for(i = arenas.begin(); i != arenas.end(); ++i) {
        delete i->second;
    }
}

hostptr_t Cache::get()
{
    ArenaMap::iterator i;
    lock();
    for(i = arenas.begin(); i != arenas.end(); ++i) {
        if(i->second->empty()) continue;
        TRACE(LOCAL,"Cache %p gets memory from arena %p", this, i->second);
        unlock();
        return i->second->get();
    }
    // There are no free objects in any arena
    Arena *arena = new Arena(manager_, mode_, objectSize);
    if(arena->valid() == false) {
        delete arena; unlock();
        return NULL;
    }
    TRACE(LOCAL,"Cache %p creates new arena %p with key %p", this, arena, arena->key());
    arenas.insert(ArenaMap::value_type(arena->key(), arena));
    hostptr_t ptr = arena->get();
    unlock();
    return ptr;
}

}}}
