#ifndef GMAC_MEMORY_PROTOCOL_LAZY_IMPL_H_
#define GMAC_MEMORY_PROTOCOL_LAZY_IMPL_H_

#include "memory/BlockGroup.h"

namespace __impl { namespace memory { namespace protocol {

template<typename T>
inline Lazy<T>::Lazy(bool eager) :
    gmac::memory::protocol::LazyBase(eager)
{}

template<typename T>
inline Lazy<T>::~Lazy()
{}

template<typename T>
memory::Object *
Lazy<T>::createObject(core::Mode &current, size_t size, hostptr_t cpuPtr,
                      GmacProtection prot, unsigned flags)
{
    gmacError_t err;
    Object *ret = new T(*this, current, cpuPtr,
                        size, LazyBase::state(prot), err);
    if(ret == NULL) return ret;
    if(err != gmacSuccess) {
        ret->decRef();
        return NULL;
    }
    if (limit_ != size_t(-1)) {
#if 0
        lock();
        LazyBase::limit_ += 2;
        unlock();
#endif
    }
    return ret;
}

}}}

#endif
