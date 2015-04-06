#include "util/Logger.h"

#include "Buddy.h"

namespace __impl { namespace util { namespace allocator {

Buddy::Buddy(hostptr_t addr, size_t size) :
    gmac::util::Lock("Buddy"),
    addr_(addr),
    size_(round((uint32_t)size)),
    index_(index(size_))
{
    _tree[index_].push_back(0);
}

Buddy::~Buddy()
{
    _tree.clear();
}

uint8_t Buddy::ones(register uint32_t x) const
{
    /* 32-bit recursive reduction using SWAR...
       but first step is mapping 2-bit values
       into sum of 2 1-bit values in sneaky way
    */
    x -= ((x >> 1) & 0x55555555);
    x = (((x >> 2) & 0x33333333) + (x & 0x33333333));
    x = (((x >> 4) + x) & 0x0f0f0f0f);
    x += (x >> 8);
    x += (x >> 16);
    return (x & 0x0000003f);
}

uint8_t Buddy::index(register uint32_t x) const
{
    register int32_t y = (x & (x - 1));

    y |= -y;
    y >>= 31;
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return uint8_t(ones(x >> 1) - y);
}

uint32_t Buddy::round(register uint32_t x) const
{
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;

    return x;
}

off_t Buddy::getFromList(uint8_t i)
{
    if(i > index_) {
        TRACE(LOCAL,"Requested size (%d) larger than available I/O memory", 1 << i);
        return -1;
    }
    /* Check for spare chunks of the requested size */
    List &list = _tree[i];
    if(list.empty() == false) {
        TRACE(LOCAL,"Returning chunk of %d bytes", 1 << i);
        off_t ret = list.front();
        list.pop_front();
        return ret;
    }

    /* No spare chunks, try splitting a bigger one */
    TRACE(LOCAL,"Asking for chunk of %d bytes (%d)", 1 << (i + 1), i + 1);
    off_t larger = getFromList(i + 1);
    if(larger == -1) return -1; /* Not enough memory */
    TRACE(LOCAL,"Spliting chunk 0x%x from size %d into two halves", larger, (1 << (i + 1)));
    off_t mid = larger + (1 << i);
    list.push_back(mid);
    return larger;
}

void Buddy::putToList(off_t addr, uint8_t i)
{
    if(i == index_) {
        _tree[i].push_back(addr);
        return;
    }
    
    /* Try merging buddies */
    long_t mask = ~((1 << (i + 1)) - 1);
    List &list = _tree[i];
    List::iterator buddy;
    for(buddy = list.begin(); buddy != list.end(); ++buddy) {
        if((*buddy & mask) != (addr & mask))
            continue;
        TRACE(LOCAL,"Merging 0x%x and 0x%x into a %d chunk", addr, *buddy, 1 << (i + 1));
        list.erase(buddy);
        return putToList((addr & mask), i + 1);        
    }
    TRACE(LOCAL,"Inserting 0x%x into %d chunk list", addr, 1 << i);
    list.push_back(addr);
    return;

}

hostptr_t Buddy::get(size_t &size)
{
    uint8_t i = index((uint32_t)size);
    uint32_t realSize = (uint32_t) 1 << i;
    TRACE(LOCAL,"Request for %d bytes of I/O memory", realSize);
    lock();
    off_t off = getFromList(i);
    unlock();
    if(off < 0) return NULL;
    TRACE(LOCAL,"Returning address at offset %d", off);
    size = realSize;
    return addr_ + off;
}

void Buddy::put(hostptr_t addr, size_t size)
{
    uint8_t i = index((uint32_t)size);
    off_t off = off_t(addr - addr_);
    TRACE(LOCAL,"Releasing %d bytes at offset %d of I/O memory", size, off);
    lock();
    putToList(off, i);
    unlock();
}

}}}
