#include "PageTable.h"

#include "core/Context.h"


namespace __impl { namespace memory {

size_t PageTable::tableShift;

PageTable::PageTable() :
	lock(LockPageTable),
	pages(1)
#ifdef USE_VM
	,_clean(false), _valid(true)
#endif
{
	tableShift = int(log2(paramPageSize));
	trace("Page Size: %zd bytes", paramPageSize);
#ifndef USE_MMAP
	trace("Table Shift: %zd bits", tableShift);
	trace("Table Size: %ld entries", (1 << dirShift) / paramPageSize);
#endif
}


PageTable::~PageTable()
{ 
//#ifndef USE_MMAP
	trace("Cleaning Page Table");
	for(unsigned i = 0; i < rootTable.size(); i++) {
		if(rootTable.present(i) == false) continue;
		deleteDirectory(rootTable.value(i));
	}
//#endif
}

void PageTable::deleteDirectory(Directory *dir)
{
	for(unsigned i = 0; i < dir->size(); i++) {
		if(dir->present(i) == false) continue;
		delete dir->value(i);
	}
	delete dir;
}

void PageTable::sync()
{
#ifdef USE_VM
	if(_valid == true) return;
	enterFunction(FuncVmSync);
    //TODO: copy dirtyBitmap back from accelerator
	_valid = true;
	exitFunction();
#endif
}


void PageTable::insert(void *host, void *acc)
{
#ifndef USE_MMAP
	sync();

	enterFunction(FuncVmAlloc);
	lock.lockWrite();
#ifdef USE_VM
	_clean = false;
#endif
	// Get the root table entry
	if(rootTable.present(entry(host, rootShift, rootTable.size())) == false) {
		rootTable.create(entry(host, rootShift, rootTable.size()));
		pages++;
	}
	Directory &dir = rootTable.get(entry(host, rootShift, rootTable.size()));

	if(dir.present(entry(host, dirShift, dir.size())) == false) {
		dir.create(entry(host, dirShift, dir.size()),
			(1 << dirShift) / paramPageSize);
		pages++;
	}
	Table &table = dir.get(entry(host, dirShift, dir.size()));

	unsigned e = entry(host, tableShift, table.size());
	assertion(table.present(e) == false || (uint8_t *)table.value(e) == acc);

	table.insert(entry(host, tableShift, table.size()), acc);
	trace("PT inserts: 0x%x -> %p", entry(host, tableShift, table.size()), acc);
	lock.unlock();
	exitFunction();
#endif
}

void PageTable::remove(void *host)
{
#ifndef USE_MMAP
	sync();
	enterFunction(FuncVmFree);
	lock.lockWrite();
#ifdef USE_VM
	_clean = false;
#endif

	if(rootTable.present(entry(host, rootShift, rootTable.size())) == false) {
		exitFunction();
		return;
	}
	Directory &dir = rootTable.get(entry(host, rootShift, rootTable.size()));
	if(dir.present(entry(host, dirShift, dir.size())) == false) {
		exitFunction();
		return;
	}
	Table &table = dir.get(entry(host, dirShift, dir.size()));
	table.remove(entry(host, tableShift, table.size()));
	lock.unlock();
	exitFunction();
#endif
}

void *PageTable::translate(void *host) 
{
#ifdef USE_MMAP
	return host;
#else
	sync();

	lock.lockRead();
	if(rootTable.present(entry(host, rootShift, rootTable.size())) == false) {
		lock.unlock();
        trace("Translate %p to NULL in RootTable");
		return NULL;
	}
	Directory &dir = rootTable.get(entry(host, rootShift, rootTable.size()));
	if(dir.present(entry(host, dirShift, dir.size())) == false) {
		lock.unlock();
        trace("Translate %p to NULL in Directory");
		return NULL;
	}
	Table &table = dir.get(entry(host, dirShift, dir.size()));
	uint8_t *addr =
		(uint8_t *)table.value(entry(host, tableShift, table.size()));
	lock.unlock();
	addr += offset(host);
    trace("Translate %p -> %p", host, (void *)addr);
	return (void *)addr;
#endif
}


}}
