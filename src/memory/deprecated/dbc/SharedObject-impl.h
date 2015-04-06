#ifndef GMAC_MEMORY_DBC_SHAREDOBJECT_IMPL_H_
#define GMAC_MEMORY_DBC_SHAREDOBJECT_IMPL_H_

namespace __dbc { namespace memory {

template<typename State>
SharedObject<State>::SharedObject(__impl::memory::Protocol &protocol, __impl::core::Mode &owner, hostptr_t addr, size_t size, typename State::ProtocolState init, gmacError_t &err) :
    __impl::memory::SharedObject<State>(protocol, owner, addr, size, init, err)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
}

template<typename State>
SharedObject<State>::~SharedObject()
{
}

}}

#endif /* SHAREDOBJECT_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
