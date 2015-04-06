#ifndef GMAC_MEMORY_OBJECTMAP_IMPL_H_
#define GMAC_MEMORY_OBJECTMAP_IMPL_H_

#include "Object.h"

namespace __impl { namespace memory {

inline
gmacError_t
ObjectMap::forEachObject(gmacError_t (Object::*f)(void))
{
    iterator i;
    lockRead();
    for(i = begin(); i != end(); ++i) {
        gmacError_t ret = (i->second->*f)();
        if(ret != gmacSuccess) {
            unlock();
            return ret;
        }
    }
    unlock();
    return gmacSuccess;
}

template <typename P1>
gmacError_t
ObjectMap::forEachObject(gmacError_t (Object::*f)(P1 &), P1 &p1)
{
    const_iterator i;
    lockRead();
    for(i = begin(); i != end(); i++) {
        gmacError_t ret = (i->second->*f)(p1);
        if(ret != gmacSuccess) {
            unlock();
            return ret;
        }
    }
    unlock();
    return gmacSuccess;
}


#ifdef DEBUG
inline
gmacError_t ObjectMap::dumpObjects(const std::string &dir, std::string prefix, protocol::common::Statistic stat) const
{
    lockRead();
    const_iterator i;
    for(i = begin(); i != end(); ++i) {
        Object &obj = *(i->second);
        std::stringstream name;
        name << dir << prefix << "#" << obj.getId() << "-" << obj.getDumps(stat) << "_" << protocol::common::StatisticName[stat];

        std::ofstream out(name.str().c_str(), std::ios_base::trunc);
        ASSERTION(out.good());
        obj.dump(out, stat);

        out.close();
    }
    unlock();
    return gmacSuccess;
}

inline
gmacError_t ObjectMap::dumpObject(const std::string &dir, std::string prefix, protocol::common::Statistic stat, hostptr_t ptr) const
{
    Object *obj = getObject(ptr, 1);
    lockRead();
    ASSERTION(obj != NULL);
    std::stringstream name;
    name << dir << prefix << "#" << obj->getId() << "-" << obj->getDumps(stat) << "_" << protocol::common::StatisticName[stat];

    std::ofstream out(name.str().c_str(), std::ios_base::trunc);
    ASSERTION(out.good());
    obj->dump(out, stat);

    out.close();
    unlock();
    return gmacSuccess;
}
#endif

inline
bool
ObjectMap::hasModifiedObjects() const
{
    lockRead();
    bool ret = modifiedObjects_;
    unlock();
    return ret;
}

inline
void
ObjectMap::invalidateObjects()
{
    lockWrite();
    modifiedObjects_ = false;
    unlock();
}

inline
void
ObjectMap::modifiedObjects_unlocked()
{
    modifiedObjects_ = true;
    releasedObjects_ = false;
}

inline
void
ObjectMap::modifiedObjects()
{
    lockWrite();
    modifiedObjects_unlocked();
    unlock();
}

inline
bool
ObjectMap::releasedObjects() const
{
    lockRead();
    bool ret = releasedObjects_;
    unlock();
    return ret;
}

inline
Protocol &
ObjectMap::getProtocol()
{
    return protocol_;
}

#ifdef USE_VM
inline vm::Bitmap&
ObjectMap::getBitmap()
{
    return bitmap_;
}

inline const vm::Bitmap&
ObjectMap::getBitmap() const
{
    return bitmap_;
}
#endif

}}

#endif /* OBJECTMAP_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
