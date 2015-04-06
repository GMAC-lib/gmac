#include "memory/Memory.h"
#include "memory/windows/FileMap.h"
#include "core/Mode.h"

namespace __impl { namespace memory {

static DWORD ProtBits[] = {
	PAGE_NOACCESS,
	PAGE_READONLY,
	PAGE_READWRITE,
	PAGE_READWRITE
};


static FileMap Files;

int Memory::protect(hostptr_t addr, size_t count, GmacProtection prot)
{
    TRACE(GLOBAL, "Setting memory permisions to %d @ %p - %p", prot, addr, addr + count);
	DWORD old = 0;
    BOOL ret = VirtualProtect(addr, count, ProtBits[prot], &old);
    ASSERTION(ret == TRUE);
    return 0;
}

hostptr_t Memory::map(hostptr_t addr, size_t count, GmacProtection prot)
{
    hostptr_t cpuAddr = NULL;

	// Create anonymous file to be mapped
	HANDLE handle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, (DWORD)count, NULL);
	if(handle == NULL) return NULL;
	
	// Create a view of the file in the previously allocated virtual memory range
	if((cpuAddr = (hostptr_t)MapViewOfFileEx(handle, FILE_MAP_WRITE, 0, 0, (DWORD)count, addr)) == NULL) {
		CloseHandle(handle);
		return NULL;
	}

	if(Files.insert(handle, cpuAddr, count) == false) {
		CloseHandle(handle);
		return NULL;
	}

	// Make sure that the range has the requested virtual protection bits
	protect(cpuAddr, count, prot);
    return cpuAddr;
}

hostptr_t Memory::shadow(hostptr_t addr, size_t count)
{
	TRACE(GLOBAL, "Getting shadow mapping for %p ("FMT_SIZE" bytes)", addr, count);
	FileMapEntry entry = Files.find(addr);
	if(entry.handle() == NULL) return NULL;
	off_t offset = off_t(addr - entry.address());
	hostptr_t ret = (hostptr_t)MapViewOfFile(entry.handle(), FILE_MAP_WRITE, 0, (DWORD)offset, (DWORD)count);
	return ret;
}

void Memory::unshadow(hostptr_t addr, size_t /*count*/)
{
	BOOL ret = UnmapViewOfFile(addr);
	ASSERTION(ret == TRUE);
}


void Memory::unmap(hostptr_t addr, size_t /*count*/)
{
	FileMapEntry entry = Files.find(addr);
	if(Files.remove(addr) == false) return;
	UnmapViewOfFile(entry.address());
	CloseHandle(entry.handle());
}

}}

