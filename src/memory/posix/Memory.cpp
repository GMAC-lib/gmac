#include "memory/Memory.h"
#include "memory/posix/FileMap.h"
#include "core/Mode.h"

#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>

namespace __impl { namespace memory {

int ProtBits[] = {
    PROT_NONE,
    PROT_READ,
    PROT_WRITE,
    PROT_READ | PROT_WRITE
};

static FileMap Files;

int Memory::protect(hostptr_t addr, size_t count, GmacProtection prot)
{
    trace::EnterCurrentFunction();
    TRACE(GLOBAL, "Setting memory permisions to %d @ %p - %p", prot, addr, addr + count);
    int ret = mprotect(addr, count, ProtBits[prot]);
    CFATAL(ret == 0);
    trace::ExitCurrentFunction();
    return 0;
}

hostptr_t Memory::map(hostptr_t addr, size_t count, GmacProtection prot)
{
    trace::EnterCurrentFunction();
    hostptr_t cpuAddr = NULL;
    char tmp[FILENAME_MAX];

    // Create new shared memory file
    snprintf(tmp, FILENAME_MAX, "/tmp/gmacXXXXXX");
    int fd = mkstemp(tmp);
    if(fd < 0) {
        trace::ExitCurrentFunction();
        return NULL;
    }
    unlink(tmp);

    if(ftruncate(fd, count) < 0) {
        close(fd);
        trace::ExitCurrentFunction();
        return NULL;
    }

    if (addr == NULL) {
        cpuAddr = hostptr_t(mmap(addr, count, ProtBits[prot], MAP_SHARED, fd, 0));
        TRACE(GLOBAL, "Getting map: %d @ %p - %p", prot, cpuAddr, addr + count);
    } else {
        cpuAddr = addr;
        if(mmap(cpuAddr, count, ProtBits[prot], MAP_SHARED | MAP_FIXED, fd, 0) != cpuAddr) {
            close(fd);
            trace::ExitCurrentFunction();
            return NULL;
        }
        TRACE(GLOBAL, "Getting fixed map: %d @ %p - %p", prot, addr, addr + count);
    }

    if(Files.insert(fd, cpuAddr, count) == false) {
        munmap(cpuAddr, count);
        close(fd);
        trace::ExitCurrentFunction();
        return NULL;
    }

    trace::ExitCurrentFunction();
    return cpuAddr;
}

hostptr_t Memory::shadow(hostptr_t addr, size_t count)
{
    trace::EnterCurrentFunction();
    TRACE(GLOBAL, "Getting shadow mapping for %p (%zd bytes)", addr, count);
    FileMapEntry entry = Files.find(addr);
    if(entry.fd() == -1) {
        trace::ExitCurrentFunction();
        return NULL;
    }
    off_t offset = off_t(addr - entry.address());
#if defined(__APPLE__)
    hostptr_t ret = hostptr_t(mmap(NULL, count, ProtBits[GMAC_PROT_READWRITE], MAP_SHARED, entry.fd(), offset));
#else
    hostptr_t ret = hostptr_t(mmap(NULL, count, ProtBits[GMAC_PROT_READWRITE], MAP_SHARED | MAP_POPULATE, entry.fd(), offset));
#endif
    trace::ExitCurrentFunction();
    return ret;
} 

void Memory::unshadow(hostptr_t addr, size_t count)
{
    trace::EnterCurrentFunction();
    ::munmap(addr, count);
    trace::ExitCurrentFunction();
}

void Memory::unmap(hostptr_t addr, size_t count)
{
    trace::EnterCurrentFunction();
    FileMapEntry entry = Files.find(addr);
    if(Files.remove(addr) == false) {
        trace::ExitCurrentFunction();
        return;
    }
    munmap(addr, count);
    close(entry.fd());
    trace::ExitCurrentFunction();
}

}}

