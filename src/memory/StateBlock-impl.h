#ifndef GMAC_MEMORY_STATEBLOCK_IMPL_H_
#define GMAC_MEMORY_STATEBLOCK_IMPL_H_

namespace __impl { namespace memory {

template<typename State>
inline StateBlock<State>::StateBlock(Protocol &protocol, hostptr_t addr,
                               hostptr_t shadow, size_t size, typename State::ProtocolState init) :
    gmac::memory::Block(protocol, addr, shadow, size),
    State(init)
{
}

template<typename State>
inline hostptr_t
StateBlock<State>::getShadow() const
{
    return Block::shadow_;
}

}}

#endif
