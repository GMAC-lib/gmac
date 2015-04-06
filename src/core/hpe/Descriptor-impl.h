#ifndef GMAC_CORE_HPE_DESCRIPTOR_IMPL_H_
#define GMAC_CORE_HPE_DESCRIPTOR_IMPL_H_

#include "util/Logger.h"

namespace __impl { namespace core { namespace hpe {

template <typename K>
inline
Descriptor<K>::Descriptor(const char * name, K key) :
    key_(key), name_(name)
{
    ASSERTION(key_ != NULL);
}

template <typename K>
inline const char *
Descriptor<K>::getName() const
{
    return name_;
}

template <typename K>
inline K
Descriptor<K>::key() const
{
    return key_;
}

}}}

#endif
