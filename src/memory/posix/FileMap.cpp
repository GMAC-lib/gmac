#include "memory/posix/FileMap.h"

namespace __impl { namespace memory {

FileMap::FileMap() :
	gmac::util::RWLock("FileMap")
{ }

FileMap::~FileMap()
{ }

bool FileMap::insert(int fd, hostptr_t address, size_t size)
{
	hostptr_t key = address + size;
	lockWrite();
	std::pair<Parent::iterator, bool> ret = Parent::insert(
		Parent::value_type(key, FileMapEntry(fd, address, size)));
	unlock();
	return ret.second;
}

bool FileMap::remove(hostptr_t address)
{
	bool ret = true;
	lockWrite();
	Parent::iterator i = Parent::upper_bound(address);
	if(i != Parent::end()) Parent::erase(i);
	else ret = false;
	unlock();
	return ret;
}

const FileMapEntry FileMap::find(hostptr_t address) const
{
	FileMapEntry ret(-1, NULL, 0);
	lockRead();
	Parent::const_iterator i = Parent::upper_bound(address);
	if(i != Parent::end()) {
		if((uint8_t *)i->second.address() <= (uint8_t *) address) {
			ret = i->second;
		}
	}
	unlock();
	return ret;
}
}}

