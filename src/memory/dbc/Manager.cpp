#ifdef USE_DBC

#include "core/IOBuffer.h"

#include "Manager.h"

namespace __dbc { namespace memory {

Manager::Manager(ProcessImpl &proc) :
    Parent(proc)
{
}

Manager::~Manager()
{
}
#if 0
gmacError_t
Manager::map(void *addr, size_t size, GmacProtection prot)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = Parent::map(addr, size, prot);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
Manager::unmap(void *addr, size_t size)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    REQUIRES(size > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = Parent::unmap(addr, size);
    // POSTCONDITIONS

    return ret;
}
#endif
gmacError_t
Manager::alloc(ModeImpl &mode, hostptr_t *addr, size_t size)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = Parent::alloc(mode, addr, size);
    // POSTCONDITIONS

    return ret;
}

#if 0
gmacError_t
Manager::globalAlloc(void **addr, size_t size, GmacGlobalMallocType hint)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = Parent::globalAlloc(addr, size, hint);
    // POSTCONDITIONS

    return ret;
}
#endif

gmacError_t
Manager::free(ModeImpl &mode, hostptr_t addr)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    // CALL IMPLEMENTATION
    gmacError_t ret = Parent::free(mode, addr);
    // POSTCONDITIONS

    return ret;
}

bool
Manager::signalRead(ModeImpl &mode, hostptr_t addr)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    // CALL IMPLEMENTATION
    bool ret = Parent::signalRead(mode, addr);
    // POSTCONDITIONS

    return ret;
}

bool
Manager::signalWrite(ModeImpl &mode, hostptr_t addr)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    // CALL IMPLEMENTATION
    bool ret = Parent::signalWrite(mode, addr);
    // POSTCONDITIONS

    return ret;
}


gmacError_t
Manager::toIOBuffer(ModeImpl &mode, IOBufferImpl &buffer, size_t bufferOff, const hostptr_t addr, size_t size)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    REQUIRES(size > 0);
    REQUIRES(size <= buffer.size() - bufferOff);
    // CALL IMPLEMENTATION
    gmacError_t ret = Parent::toIOBuffer(mode, buffer, bufferOff, addr, size);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
Manager::fromIOBuffer(ModeImpl &mode, hostptr_t addr, IOBufferImpl &buffer, size_t bufferOff, size_t size)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    REQUIRES(size > 0);
    REQUIRES(size <= buffer.size() - bufferOff);
    // CALL IMPLEMENTATION
    gmacError_t ret = Parent::fromIOBuffer(mode, addr, buffer, bufferOff, size);
    // POSTCONDITIONS

    return ret;
}
gmacError_t
Manager::memcpy(ModeImpl &mode, hostptr_t dst, const hostptr_t src, size_t n)
{
    // PRECONDITIONS
    REQUIRES(src != NULL);
    REQUIRES(dst != NULL);
    REQUIRES(n > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = Parent::memcpy(mode, dst, src, n);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
Manager::memset(ModeImpl &mode, hostptr_t dst, int c, size_t n)
{
    // PRECONDITIONS
    REQUIRES(dst != NULL);
    REQUIRES(n > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = Parent::memset(mode, dst, c, n);
    // POSTCONDITIONS

    return ret;
}

}}

#endif
