#ifdef USE_DBC

#include "core/IOBuffer.h"
#include "memory/Block.h"

namespace __dbc { namespace memory {

Block::Block(__impl::memory::Protocol &protocol, hostptr_t addr, hostptr_t shadow, size_t size) :
    __impl::memory::Block(protocol, addr, shadow, size)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    REQUIRES(addr != NULL);
    REQUIRES(shadow != NULL);
}

Block::~Block()
{
}

gmacError_t
Block::memoryOp(__impl::memory::Protocol::MemoryOp op, __impl::core::IOBuffer &buffer, size_t size, size_t bufferOffset, size_t blockOffset)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    REQUIRES(blockOffset + size <= size_);
    REQUIRES(bufferOffset + size <= buffer.size());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::Block::memoryOp(op, buffer, size, bufferOffset, blockOffset);
    // POSTCONDITIONS
    
    return ret;
}


gmacError_t
Block::memset(int v, size_t size, size_t blockOffset)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    REQUIRES(blockOffset + size <= size_);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::Block::memset(v, size, blockOffset);
    // POSTCONDITIONS
    
    return ret;
}


}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
