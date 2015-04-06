#ifndef GMAC_CORE_ALLOCATIONMAP_IMPL_H_
#define GMAC_CORE_ALLOCATIONMAP_IMPL_H_

#include "config/common.h"
#include "util/Logger.h"

namespace __impl { namespace core { 

inline
AllocationMap::AllocationMap() :
    gmac::util::RWLock("AllocationMap")
{
}

inline
void AllocationMap::insert(hostptr_t key, accptr_t val, size_t size)
{
    lockWrite();
    ASSERTION(MapAlloc::find(key) == end());
    MapAlloc::insert(MapAlloc::value_type(key, PairAlloc(val, size)));
    unlock();
}

inline
void AllocationMap::erase(hostptr_t key, size_t size)
{
    lockWrite();
    MapAlloc::erase(key);
    unlock();
}

inline
bool AllocationMap::find(hostptr_t key, accptr_t &val, size_t &size)
{
    lockRead();
    MapAlloc::const_iterator it = MapAlloc::find(key);
    bool ret = false;
    if (it != MapAlloc::end()) {
        val  = it->second.first;
        size = it->second.second;
        ret = true;
    }
    unlock();
    return ret;
}


}}

#endif
