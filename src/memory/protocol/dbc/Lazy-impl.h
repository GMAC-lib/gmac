#ifndef GMAC_MEMORY_PROTOCOL_DBC_LAZY_IMPL_H_
#define GMAC_MEMORY_PROTOCOL_DBC_LAZY_IMPL_H_

#include "memory/protocol/lazy/LazyTypes.h"

namespace __dbc { namespace memory { namespace protocol {

template <typename T>
Lazy<T>::Lazy(bool eager) :
    Parent(eager)
{
}

template <typename T>
Lazy<T>::~Lazy()
{
}


}}}

#endif //GMAC_MEMORY_PROTOCOL_DBC_LAZY_IMPL_H_

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
