#if defined(USE_DBC)

#include "Buddy.h"

namespace __dbc { namespace util { namespace allocator {

Buddy::Buddy(hostptr_t addr, size_t size) :
    __impl::util::allocator::Buddy(addr, size)
{}

off_t Buddy::getFromList(uint8_t i)
{
    return __impl::util::allocator::Buddy::getFromList(i);
}

void Buddy::putToList(off_t addr, uint8_t i)
{
    REQUIRES(addr >= 0 && size_t(addr) < size_);
    return __impl::util::allocator::Buddy::putToList(addr, i);
}

hostptr_t Buddy::get(size_t &size)
{
    REQUIRES(size > 0);
    hostptr_t ret = __impl::util::allocator::Buddy::get(size);
    ENSURES(ret == NULL || (ret >= addr_ && ret <= (addr_ + size_ - size)));
    return ret;
}

void Buddy::put(hostptr_t addr, size_t size)
{
    REQUIRES(addr >= addr_ && addr <= (addr_ + size_ - size));
    REQUIRES(size > 0);
    __impl::util::allocator::Buddy::put(addr, size);
}

}}}

#endif
