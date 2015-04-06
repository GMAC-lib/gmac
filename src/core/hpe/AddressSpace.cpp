#include "Process.h"
#include "Mode.h"
#include "Context.h"

#include "memory/ObjectMap.h"
#include "memory/Object.h"

namespace __impl { namespace core { namespace hpe {

AddressSpace::AddressSpace(const char *name, Process &parent) :
    Parent(name), parent_(parent)
{
}

AddressSpace::~AddressSpace()
{
    TRACE(LOCAL,"Cleaning Memory AddressSpace");
    //TODO: actually clean the memory map
}

memory::Object *
AddressSpace::getObject(const hostptr_t addr, size_t size) const
{
    memory::Object *ret = NULL;
    // Lookup in the current map
    ret = mapFind(addr, size);

    // Exit if we have found it
    if(ret != NULL) goto exit_func;

    // Check global maps
    {
        ret = parent_.shared().mapFind(addr, size);
        if (ret != NULL) goto exit_func;
        ret = parent_.global().mapFind(addr, size);
        if (ret != NULL) goto exit_func;
        ret = parent_.orphans().mapFind(addr, size);
        if (ret != NULL) goto exit_func;
    }

exit_func:
    if(ret != NULL) ret->incRef();
    return ret;
}

bool AddressSpace::addObject(memory::Object &obj)
{
    TRACE(LOCAL,"Adding Shared Object %p", obj.addr());
    bool ret = Parent::addObject(obj);
    if(ret == false) return ret;
    memory::ObjectMap &shared = parent_.shared();
    ret = shared.addObject(obj);
    return ret;
}


bool AddressSpace::removeObject(memory::Object &obj)
{
    bool ret = Parent::removeObject(obj);
    hostptr_t addr = obj.addr();
    // Shared object
    memory::ObjectMap &shared = parent_.shared();
    ret = shared.removeObject(obj);
    if(ret == true) {
        TRACE(LOCAL,"Removed Shared Object %p", addr);
        return true;
    }

    // Replicated object
    memory::ObjectMap &global = parent_.global();
    ret = global.removeObject(obj);
    if(ret == true) {
        TRACE(LOCAL,"Removed Global Object %p", addr);
        return true;
    }

    // Orphan object
    memory::ObjectMap &orphans = parent_.orphans();
    ret = orphans.removeObject(obj);
    if(ret == true) {
        TRACE(LOCAL,"Removed Orphan Object %p", addr);
    }

    return ret;
}

void AddressSpace::addOwner(Process &proc, Mode &mode)
{
    memory::ObjectMap &global = proc.global();
    iterator i;
    global.lockWrite();
    for(i = global.begin(); i != global.end(); ++i) {
        i->second->addOwner(mode);
    }
    global.unlock();
}

void AddressSpace::removeOwner(Process &proc, Mode &mode)
{
    memory::ObjectMap &global = proc.global();
    iterator i;
    global.lockWrite();
    for (i = global.begin(); i != global.end(); ++i) {
        i->second->removeOwner(mode);
    }
    global.unlock();

    /*
    memory::ObjectMap &shared = proc.shared();
    iterator j;
    shared.lockWrite();
    for (j = shared.begin(); j != shared.end(); j++) {
        // Owners already removed in Mode::cleanUp
        j->second->removeOwner(mode);
    }
    shared.unlock();
    */
}

}}}
