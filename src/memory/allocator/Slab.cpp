#include "Slab.h"

#include "config/common.h"
#include "core/Mode.h"

namespace __impl { namespace memory { namespace allocator {

Cache &Slab::createCache(core::Mode &mode, CacheMap &map, long_t key, size_t size)
{
    Cache *cache = new __impl::memory::allocator::Cache(manager_, mode, size);
    std::pair<CacheMap::iterator, bool> ret = map.insert(CacheMap::value_type(key, cache));
    ASSERTION(ret.second == true);
    return *cache;
}

Cache &Slab::get(core::Mode &current, long_t key, size_t size)
{
    CacheMap *map = NULL;
    ModeMap::iterator i;
    modes.lockRead();
    i = modes.find(&current);
    if(i != modes.end()) map = &(i->second);
    modes.unlock();
    if(map == NULL) {
        modes.lockWrite();
        Cache &ret = createCache(current, modes[&current], key, size);
        modes.unlock();
        return ret;
    }
    else {
        CacheMap::iterator j;
        j = map->find(key);
        if(j == map->end())
            return createCache(current, *map, key, size);
        else
            return *j->second;
    }
}

void Slab::cleanup(core::Mode &current)
{
    ModeMap::iterator i;
    modes.lockRead();
    i = modes.find(&current);
    modes.unlock();
    if(i == modes.end()) return;
    CacheMap::iterator j;
    for(j = i->second.begin(); j != i->second.end(); ++j) {
        delete j->second;
    }
    i->second.clear();
    modes.lockWrite();
    modes.erase(i);
    modes.unlock();
}

hostptr_t Slab::alloc(core::Mode &current, size_t size, hostptr_t addr)
{
    Cache &cache = get(current, long_t(addr), size);
    TRACE(LOCAL,"Using cache %p", &cache);
    hostptr_t ret = cache.get();
    if(ret == NULL) return NULL;
    addresses.lockWrite();
    addresses.insert(AddressMap::value_type(ret, &cache));
    addresses.unlock();
    TRACE(LOCAL,"Retuning address %p", ret);
    return ret;
}

bool Slab::free(core::Mode &current, hostptr_t addr)
{
    addresses.lockWrite();
    AddressMap::iterator i = addresses.find(addr);
    if(i == addresses.end()) {
        addresses.unlock();
        TRACE(LOCAL,"%p was not delivered by slab allocator", addr); 
        return false;
    }
    TRACE(LOCAL,"Inserting %p in cache %p", addr, i->second);
    Cache &cache = *(i->second);
    addresses.erase(i);
    addresses.unlock();
    cache.put(addr);
    return true;
}

}}}
