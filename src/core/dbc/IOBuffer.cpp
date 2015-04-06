#ifdef USE_DBC

#include "core/dbc/IOBuffer.h"

namespace __dbc { namespace core {


IOBuffer::IOBuffer(void *addr, size_t size, bool async, GmacProtection prot) :
    __impl::core::IOBuffer(addr, size, async, prot)
{
    // This check goes out because OpenCL will always use 0 as base address
    REQUIRES(addr != NULL);
    REQUIRES(size > 0);
}

IOBuffer::~IOBuffer()
{
}

uint8_t *IOBuffer::addr() const
{
    uint8_t *ret = __impl::core::IOBuffer::addr();
    ENSURES(ret != NULL);
    return ret;
}

uint8_t *IOBuffer::end() const
{
    uint8_t *ret = __impl::core::IOBuffer::end();
    ENSURES(ret !=  NULL);
    return ret;
}

size_t IOBuffer::size() const
{
    size_t ret = __impl::core::IOBuffer::size();
    ENSURES(ret > 0);
    return ret;
}

}}
#endif 

