#ifndef GMAC_MEMORY_ALLOCATOR_SLAB_IPP_
#define GMAC_MEMORY_ALLOCATOR_SLAB_IPP_

namespace __impl { namespace memory { namespace allocator {

inline Slab::Slab(Manager &manager) : manager_(manager) {}

inline Slab::~Slab()
{
    ModeMap::iterator i;
    modes.lockWrite();
    for(i = modes.begin(); i != modes.end(); ++i) {
        CacheMap &map = i->second;
        CacheMap::iterator j;
        for(j = map.begin(); j != map.end(); ++j) {
            delete j->second;
        }
        map.clear();
    }
    modes.clear();
    modes.unlock();
}

}}}
#endif
