#ifndef GMAC_MEMORY_HOSTMAPPEDOBJECT_IMPL_H_
#define GMAC_MEMORY_HOSTMAPPEDOBJECT_IMPL_H_

namespace __impl { namespace memory {

inline HostMappedSet::HostMappedSet() :
    RWLock("HostMappedSet")
{}

inline HostMappedSet::~HostMappedSet()
{}

#ifdef USE_OPENCL
inline gmacError_t
HostMappedObject::acquire(core::Mode &current)
{
    gmacError_t ret = gmacSuccess;
    ret = current.acquire(addr_);
    return ret;
}

inline gmacError_t
HostMappedObject::release(core::Mode &current)
{
    gmacError_t ret = gmacSuccess;
    ret = current.release(addr_);
    return ret;
}
#endif

inline hostptr_t HostMappedObject::addr() const
{
    return addr_;
}

inline size_t HostMappedObject::size() const
{
    return size_;
}

inline void HostMappedObject::remove(hostptr_t addr)
{
    set_.remove(addr);
}

inline HostMappedObject *HostMappedObject::get(const hostptr_t addr)
{
    return set_.get(addr);
}

}}

#endif
