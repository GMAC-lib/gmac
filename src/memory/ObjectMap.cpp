#include "core/Mode.h"
#include "util/FileSystem.h"

#include "ObjectMap.h"
#include "Object.h"
#include "Protocol.h"

namespace __impl { namespace memory {

#ifdef DEBUG
Atomic ObjectMap::StatsInit_ = 0;
Atomic ObjectMap::StatDumps_ = 0;
std::string ObjectMap::StatsDir_ = "";

void
ObjectMap::statsInit()
{
    if (__impl::util::params::ParamStats) {
        PROCESS_T pid = __impl::util::GetProcessId();

        std::stringstream ss(std::stringstream::out);
#if defined(_MSC_VER)
        char tmpDir[256];
        GetTempPath(256, tmpDir);
        ss << tmpDir << "\\" << pid << "\\";
#else
        ss << ".gmac-" << pid << "/";
#endif
        bool created = __impl::util::MakeDir(ss.str());
        ASSERTION(created == true);
        StatsDir_ = ss.str();
    }
}
#endif

Object *
ObjectMap::mapFind(const hostptr_t addr, size_t size) const
{
    ObjectMap::const_iterator i;
    Object *ret = NULL;
    lockRead();
    const uint8_t *limit = (const uint8_t *)addr + size;
    i = upper_bound(addr);
    if(i != end() && i->second->addr() <= limit) ret = i->second;
    unlock();
    return ret;
}

ObjectMap::ObjectMap(const char *name) :
    gmac::util::RWLock(name),
    protocol_(*ProtocolInit(0)),
    modifiedObjects_(false),
    releasedObjects_(false)
#ifdef USE_VM
    , bitmap_(*this)
#endif
{

#ifdef DEBUG
    if(AtomicTestAndSet(StatsInit_, 0, 1) == 0) statsInit();
#endif
}

ObjectMap::~ObjectMap()
{
}

void
ObjectMap::cleanUp()
{
    const_iterator i;
    lockRead();
    for(i = begin(); i != end(); ++i) {
        // Decrement reference count of pointed objects to allow later destruction
        i->second->decRef();
    }
    unlock();
}

size_t ObjectMap::size() const
{
    lockRead();
    size_t ret = Parent::size();
    unlock();
    return ret;
}

bool ObjectMap::addObject(Object &obj)
{
    lockWrite();
    TRACE(LOCAL, "Insert object: %p", obj.addr());
    std::pair<iterator, bool> ret = Parent::insert(value_type(obj.end(), &obj));
    if(ret.second == true) obj.incRef();
    unlock();
    modifiedObjects_unlocked();
    return ret.second;
}

bool ObjectMap::removeObject(Object &obj)
{
    lockWrite();
    iterator i = find(obj.end());
    bool ret = (i != end());
    if(ret == true) {
#if defined(DEBUG)
        if (__impl::util::params::ParamStats) {
            unsigned dump = AtomicInc(StatDumps_);
            std::stringstream ss(std::stringstream::out);
            ss << dump << "-" << "remove";

            dumpObject(StatsDir_, ss.str(), memory::protocol::common::PAGE_FAULTS_READ, obj.addr());
            dumpObject(StatsDir_, ss.str(), memory::protocol::common::PAGE_FAULTS_WRITE, obj.addr());
            dumpObject(StatsDir_, ss.str(), memory::protocol::common::PAGE_TRANSFERS_TO_HOST, obj.addr());
            dumpObject(StatsDir_, ss.str(), memory::protocol::common::PAGE_TRANSFERS_TO_ACCELERATOR, obj.addr());
        }
#endif

        TRACE(LOCAL, "Remove object: %p", obj.addr());
        obj.decRef();
        Parent::erase(i);
    } else {
        TRACE(LOCAL, "CANNOT Remove object: %p from map with "FMT_SIZE" elems", obj.addr(), Parent::size());
    }
    unlock();
    return ret;
}

bool ObjectMap::hasObject(Object &obj) const
{
    Object *ret = NULL;
    ret = mapFind(obj.addr(), obj.size());
    return ret == &obj;
}

Object *ObjectMap::getObject(const hostptr_t addr, size_t size) const
{
    Object *ret = NULL;
    ret = mapFind(addr, size);
    if(ret != NULL) ret->incRef();
    return ret;
}

size_t ObjectMap::memorySize() const
{
    size_t total = 0;
    const_iterator i;
    lockRead();
    for(i = begin(); i != end(); ++i) {
        total += i->second->size();
    }
    unlock();
    return total;
}

gmacError_t ObjectMap::releaseObjects()
{
    lockWrite();
#ifdef DEBUG
    if (__impl::util::params::ParamStats) {
        unsigned dump = AtomicInc(StatDumps_);
        std::stringstream ss(std::stringstream::out);
        ss << dump << "-" << "release";

        dumpObjects(StatsDir_, ss.str(), memory::protocol::common::PAGE_FAULTS_READ);
        dumpObjects(StatsDir_, ss.str(), memory::protocol::common::PAGE_FAULTS_WRITE);
        dumpObjects(StatsDir_, ss.str(), memory::protocol::common::PAGE_TRANSFERS_TO_HOST);
        dumpObjects(StatsDir_, ss.str(), memory::protocol::common::PAGE_TRANSFERS_TO_ACCELERATOR);
    }
#endif

    releasedObjects_ = true;
    unlock();
    return gmacSuccess;
}

gmacError_t
ObjectMap::acquireObjects()
{
    lockWrite();
    modifiedObjects_ = false;
    releasedObjects_ = false;
    unlock();
    return gmacSuccess;
}


}}
