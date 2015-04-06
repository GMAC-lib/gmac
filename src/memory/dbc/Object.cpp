#ifdef USE_DBC

#include "core/IOBuffer.h"
#include "memory/Object.h"

namespace __dbc { namespace memory {

Object::Object(hostptr_t addr, size_t size) :
    __impl::memory::Object(addr, size)
{
}

Object::~Object()
{
}

gmacError_t
Object::memoryOp(__impl::memory::Protocol::MemoryOp op, __impl::core::IOBuffer &buffer, size_t size, size_t bufferOffset, size_t objectOffset)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    REQUIRES(objectOffset + size <= size_);
    REQUIRES(bufferOffset + size <= buffer.size());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::Object::memoryOp(op, buffer, size, bufferOffset, objectOffset);
    // POSTCONDITIONS
    
    return ret;

}


ssize_t
Object::blockBase(size_t offset) const
{
    // PRECONDITIONS
    REQUIRES(offset <= size_);
    // CALL IMPLEMENTATION
    ssize_t ret = __impl::memory::Object::blockBase(offset);
    // POSTCONDITIONS
    
    return ret;
}

size_t
Object::blockEnd(size_t offset) const
{
    // PRECONDITIONS
    REQUIRES(offset <= size_);
    // CALL IMPLEMENTATION
    size_t ret = __impl::memory::Object::blockEnd(offset);
    // POSTCONDITIONS
    
    return ret;
}

gmacError_t
Object::signalRead(hostptr_t addr)
{
    // PRECONDITIONS
    REQUIRES(addr >= addr_);
    REQUIRES(addr  < addr_ + size_);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::Object::signalRead(addr);
    // POSTCONDITIONS
    
    return ret;
}

gmacError_t
Object::signalWrite(hostptr_t addr)
{
    // PRECONDITIONS
    REQUIRES(addr >= addr_);
    REQUIRES(addr  < addr_ + size_);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::Object::signalWrite(addr);
    // POSTCONDITIONS
    
    return ret;
}

gmacError_t
Object::copyToBuffer(__impl::core::IOBuffer &buffer, size_t size, 
        size_t bufferOffset, size_t objectOffset)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    REQUIRES(objectOffset + size <= size_);
    REQUIRES(bufferOffset + size <= buffer.size());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::Object::copyToBuffer(buffer, size, bufferOffset, objectOffset);
    // POSTCONDITIONS
    
    return ret;
}

gmacError_t
Object::copyFromBuffer(__impl::core::IOBuffer &buffer, size_t size, 
        size_t bufferOffset, size_t objectOffset)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    REQUIRES(objectOffset + size <= size_);
    REQUIRES(bufferOffset + size <= buffer.size());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::Object::copyFromBuffer(buffer, size, bufferOffset, objectOffset);
    // POSTCONDITIONS
    
    return ret;
}

gmacError_t
Object::memset(size_t offset, int v, size_t size)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    REQUIRES(offset + size <= size_);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::Object::memset(offset, v, size);
    // POSTCONDITIONS
    
    return ret;

}

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
