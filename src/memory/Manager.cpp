#include "core/IOBuffer.h"
#include "core/Mode.h"
#include "core/Process.h"

#include "memory/Handler.h"
#include "memory/HostMappedObject.h"
#include "memory/Manager.h"
#include "memory/Object.h"

using __impl::util::params::ParamAutoSync;


namespace __impl { namespace memory {

ListAddr AllAddresses;

Manager::Manager(core::Process &proc) :
    proc_(proc)
{
    TRACE(LOCAL,"Memory manager starts");
    Init();
    Handler::setManager(*this);
}

Manager::~Manager()
{
}

gmacError_t
Manager::map(core::Mode &mode, hostptr_t *addr, size_t size, int flags)
{
    FATAL("MAP NOT IMPLEMENTED YET");
    gmacError_t ret = gmacSuccess;

    TRACE(LOCAL, "New mapping");
    trace::EnterCurrentFunction();
    // For integrated accelerators we want to use Centralized objects to avoid memory transfers
    // TODO: ask process instead
    // if (mode.getAccelerator().integrated()) return hostMappedAlloc(addr, size);

    memory::ObjectMap &map = mode.getAddressSpace();

    Object *object;
    if (*addr != NULL) {
        object = map.getObject(*addr);
        if(object != NULL) {
            // TODO: Remove this limitation
            ASSERTION(object->size() == size);
            ret = object->addOwner(mode);
            goto done;
        }
    }

    // Create new shared object. We set the memory as invalid to avoid stupid data transfers
    // to non-initialized objects
    object = map.getProtocol().createObject(mode, size, NULL, GMAC_PROT_READ, 0);
    if(object == NULL) {
        trace::ExitCurrentFunction();
        return gmacErrorMemoryAllocation;
    }
    object->addOwner(mode);
    *addr = object->addr();

    Memory::protect(*addr, size, GMAC_PROT_READ);

    // Insert object into memory maps
    map.addObject(*object);

done:
    object->decRef();
    trace::ExitCurrentFunction();
    return ret;
}


gmacError_t
Manager::remap(core::Mode &mode, hostptr_t old_addr, hostptr_t *new_addr, size_t new_size, int flags)
{
    FATAL("MAP NOT IMPLEMENTED YET");
    gmacError_t ret = gmacSuccess;

    TRACE(LOCAL, "New remapping");
    trace::EnterCurrentFunction();

    return ret;
}

gmacError_t
Manager::unmap(core::Mode &mode, hostptr_t addr, size_t size)
{
    FATAL("UNMAP NOT IMPLEMENTED YET");
    TRACE(LOCAL, "Unmap allocation");
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;

    memory::ObjectMap &map = mode.getAddressSpace();

    Object *object = map.getObject(addr);
    if(object != NULL)  {
        object->removeOwner(mode);
        map.removeObject(*object);
        object->decRef();
    } else {
        HostMappedObject *hostMappedObject = HostMappedObject::get(addr);
        if(hostMappedObject == NULL) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }
        hostMappedObject->decRef();
        // We need to release the object twice to effectively destroy it
        HostMappedObject::remove(addr);
        hostMappedObject->decRef();
    }
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Manager::alloc(core::Mode &mode, hostptr_t *addr, size_t size)
{
    TRACE(LOCAL, "New allocation");
    trace::EnterCurrentFunction();
    // For integrated accelerators we want to use Centralized objects to avoid memory transfers
    // TODO: ask process instead
    if (mode.hasIntegratedMemory()) {
        gmacError_t ret = hostMappedAlloc(mode, addr, size);
        trace::ExitCurrentFunction();
        return ret;
    }

    memory::ObjectMap &map = mode.getAddressSpace();

    // Create new shared object. We set the memory as invalid to avoid stupid data transfers
    // to non-initialized objects
    Object *object = map.getProtocol().createObject(mode, size, NULL, GMAC_PROT_READ, 0);
    if(object == NULL) {
        trace::ExitCurrentFunction();
        return gmacErrorMemoryAllocation;
    }
    gmacError_t ret = object->addOwner(mode);
    if (ret == gmacSuccess) {
        *addr = object->addr();
        Memory::protect(*addr, size, GMAC_PROT_READ);
        // Insert object into memory maps
        map.addObject(*object);
    }
    object->decRef();
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Manager::hostMappedAlloc(core::Mode &mode, hostptr_t *addr, size_t size)
{
    TRACE(LOCAL, "New host-mapped allocation");
    trace::EnterCurrentFunction();
    HostMappedObject *object = new HostMappedObject(mode, size);
    *addr = object->addr();
    if(*addr == NULL) {
        object->decRef();
        trace::ExitCurrentFunction();
        return gmacErrorMemoryAllocation;
    }
    trace::ExitCurrentFunction();
    return gmacSuccess;
}

gmacError_t Manager::globalAlloc(core::Mode &mode, hostptr_t *addr, size_t size, GmacGlobalMallocType hint)
{
    TRACE(LOCAL, "New global allocation");
    trace::EnterCurrentFunction();

    // If a centralized object is requested, try creating it
    if(hint == GMAC_GLOBAL_MALLOC_CENTRALIZED) {
        gmacError_t ret = hostMappedAlloc(mode, addr, size);
        if(ret == gmacSuccess) {
            trace::ExitCurrentFunction();
            return ret;
        }
    }
    Protocol *protocol = proc_.getProtocol();
    if(protocol == NULL) return gmacErrorInvalidValue;
    Object *object = protocol->createObject(mode, size, NULL, GMAC_PROT_NONE, 0);
    *addr = object->addr();
    if(*addr == NULL) {
        object->decRef();
        trace::ExitCurrentFunction();
        return hostMappedAlloc(mode, addr, size); // Try using a host mapped object
    }
    gmacError_t ret = proc_.globalMalloc(*object);
    object->decRef();
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Manager::free(core::Mode &mode, hostptr_t addr)
{
    TRACE(LOCAL, "Free allocation");
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;

    memory::ObjectMap &map = mode.getAddressSpace();

    Object *object = map.getObject(addr);
    if(object != NULL)  {
        map.removeObject(*object);
        object->decRef();
    } else {
        HostMappedObject *hostMappedObject = HostMappedObject::get(addr);
        if(hostMappedObject == NULL) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }
        hostMappedObject->decRef();
        // We need to release the object twice to effectively destroy it
        HostMappedObject::remove(addr);
        hostMappedObject->decRef();
    }
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t
Manager::getAllocSize(core::Mode &mode, const hostptr_t addr, size_t &size) const
{
    gmacError_t ret = gmacSuccess;
    trace::EnterCurrentFunction();

    memory::ObjectMap &map = mode.getAddressSpace();

    Object *obj = map.getObject(addr);
    if (obj == NULL) {
        HostMappedObject *hostMappedObject = HostMappedObject::get(addr);
        if (hostMappedObject != NULL) {
            size = hostMappedObject->size();
        } else {
            ret = gmacErrorInvalidValue;
        }
    } else {
        size = obj->size();
        obj->decRef();
    }
    trace::ExitCurrentFunction();
    return ret;
}



accptr_t
Manager::translate(core::Mode &mode, const hostptr_t addr)
{
    trace::EnterCurrentFunction();
    accptr_t ret = proc_.translate(addr);
    if(ret == 0) {
        HostMappedObject *object = HostMappedObject::get(addr);
        if(object != NULL) {
            ret = object->acceleratorAddr(mode, addr);
            object->decRef();
        }
    }
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t
Manager::acquireObjects(core::Mode &mode, const ListAddr &addrs)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;

    memory::ObjectMap &map = mode.getAddressSpace();
    if (addrs.size() == 0) {
        if (map.hasModifiedObjects() && map.releasedObjects()) {
            TRACE(LOCAL,"Acquiring Objects");
            GmacProtection prot = GMAC_PROT_READWRITE;
            ret = map.forEachObject<GmacProtection>(&Object::acquire, prot);
            map.acquireObjects();
        }
    } else {
        TRACE(LOCAL,"Acquiring call Objects");
        std::list<ObjectInfo>::const_iterator it;
        for (it = addrs.begin(); it != addrs.end(); ++it) {
            Object *obj = map.getObject(it->first);
            if (obj == NULL) {
                HostMappedObject *hostMappedObject = HostMappedObject::get(it->first);
                ASSERTION(hostMappedObject != NULL, "Address not found");
#ifdef USE_OPENCL
                hostMappedObject->acquire(mode);
#endif
                hostMappedObject->decRef();
            } else {
                GmacProtection prot = it->second;
                ret = obj->acquire(prot);
                ASSERTION(ret == gmacSuccess);
                obj->decRef();
            }
        }
    }
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t
Manager::releaseObjects(core::Mode &mode, const ListAddr &addrs)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;

    memory::ObjectMap &map = mode.getAddressSpace();
    if (addrs.size() == 0) { // Release all objects
        TRACE(LOCAL,"Releasing Objects");
        if (map.hasModifiedObjects()) {
            // Mark objects as released
            ret = map.forEachObject(&Object::release);
            ASSERTION(ret == gmacSuccess);
            // Flush protocols
            // 1. Mode protocol
            ret = map.getProtocol().releaseAll();
            ASSERTION(ret == gmacSuccess);
            // 2. Process protocol
            if (proc_.getProtocol() != NULL) {
                ret = proc_.getProtocol()->releaseAll();
                ASSERTION(ret == gmacSuccess);
            }
            map.releaseObjects();
        }
    } else { // Release given objects
        TRACE(LOCAL,"Releasing call Objects");
        ListAddr::const_iterator it;
        for (it = addrs.begin(); it != addrs.end(); ++it) {
            Object *obj = map.getObject(it->first);
            if (obj == NULL) {
                HostMappedObject *hostMappedObject = HostMappedObject::get(it->first);
                ASSERTION(hostMappedObject != NULL, "Address not found");
#ifdef USE_OPENCL
                hostMappedObject->release(mode);
#endif
                hostMappedObject->decRef();
            } else {
                // Release all the blocks in the object
                ret = obj->releaseBlocks();
                ASSERTION(ret == gmacSuccess);
                obj->decRef();
            }
        }

        // Notify protocols
        // 1. Mode protocol
        ret = map.getProtocol().releasedAll();
        ASSERTION(ret == gmacSuccess);
        // 2. Process protocol
        if (proc_.getProtocol() != NULL) {
            ret = proc_.getProtocol()->releasedAll();
            ASSERTION(ret == gmacSuccess);
        }
        map.releaseObjects();
    }
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Manager::toIOBuffer(core::Mode &mode, core::IOBuffer &buffer, size_t bufferOff, const hostptr_t addr, size_t count)
{
    if (count > (buffer.size() - bufferOff)) return gmacErrorInvalidSize;
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;
    size_t off = 0;
    do {
        // Check if the address range belongs to one GMAC object
        core::Mode * mode = proc_.owner(addr + off);
        if (mode == NULL) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }

#ifdef USE_VM
        CFATAL(mode->releasedObjects() == false, "Acquiring bitmap on released objects");
        vm::Bitmap &bitmap = mode->getBitmap();
        if (bitmap.isReleased()) {
            bitmap.acquire();
            mode->forEachObject(&Object::acquireWithBitmap);
        }
#endif

        memory::ObjectMap &map = mode->getAddressSpace();
        Object *obj = map.getObject(addr + off);
        if (!obj) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }
        // Compute sizes for the current object
        size_t objCount = obj->addr() + obj->size() - (addr + off);
        size_t c = objCount <= count - off? objCount: count - off;
        size_t objOff = addr - obj->addr();
        // Handle objects with no memory in the accelerator
        ret = obj->copyToBuffer(buffer, c, bufferOff + off, objOff);
        obj->decRef();
        if(ret != gmacSuccess) {
            trace::ExitCurrentFunction();
            return ret;
        }
        off += objCount;
        TRACE(LOCAL,"Copying from obj %p: "FMT_SIZE" of "FMT_SIZE, obj->addr(), c, count);
    } while(addr + off < addr + count);
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Manager::fromIOBuffer(core::Mode &mode, hostptr_t addr, core::IOBuffer &buffer, size_t bufferOff, size_t count)
{
    if (count > (buffer.size() - bufferOff)) return gmacErrorInvalidSize;
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;
    size_t off = 0;
    do {
        // Check if the address range belongs to one GMAC object
        core::Mode *mode = proc_.owner(addr + off);
        if (mode == NULL) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }
#ifdef USE_VM
        CFATAL(mode->releasedObjects() == false, "Acquiring bitmap on released objects");
        vm::Bitmap &bitmap = mode->getBitmap();
        if (bitmap.isReleased()) {
            bitmap.acquire();
            mode->forEachObject(&Object::acquireWithBitmap);
        }
#endif
        memory::ObjectMap &map = mode->getAddressSpace();
        Object *obj = map.getObject(addr + off);
        if (!obj) {
            trace::ExitCurrentFunction();
            return gmacErrorInvalidValue;
        }
        // Compute sizes for the current object
        size_t objCount = obj->addr() + obj->size() - (addr + off);
        size_t c = objCount <= count - off? objCount: count - off;
        size_t objOff = addr - obj->addr();
        ret = obj->copyFromBuffer(buffer, c, bufferOff + off, objOff);
        obj->decRef();
        if(ret != gmacSuccess) {
            trace::ExitCurrentFunction();
            return ret;
        }
        off += objCount;
        TRACE(LOCAL,"Copying to obj %p: "FMT_SIZE" of "FMT_SIZE, obj->addr(), c, count);
    } while(addr + off < addr + count);
    trace::ExitCurrentFunction();
    return ret;
}

bool
Manager::signalRead(core::Mode &mode, hostptr_t addr)
{
    trace::EnterCurrentFunction();
    memory::ObjectMap &map = mode.getAddressSpace();

#ifdef USE_VM
    CFATAL(map.releasedObjects() == false, "Acquiring bitmap on released objects");
    vm::Bitmap &bitmap = map.getBitmap();
    if (bitmap.isReleased()) {
        bitmap.acquire();
        map.forEachObject(&Object::acquireWithBitmap);
    }
#endif

    bool ret = true;
    Object *obj = map.getObject(addr);
    if(obj == NULL) {
        trace::ExitCurrentFunction();
        return false;
    }
    TRACE(LOCAL,"Read access for object %p: %p", obj->addr(), addr);
    gmacError_t err = obj->signalRead(addr);
    ASSERTION(err == gmacSuccess);
    obj->decRef();
    trace::ExitCurrentFunction();
    return ret;
}

bool
Manager::signalWrite(core::Mode &mode, hostptr_t addr)
{
    trace::EnterCurrentFunction();
    bool ret = true;
    memory::ObjectMap &map = mode.getAddressSpace();

#ifdef USE_VM
    CFATAL(map.releasedObjects() == false, "Acquiring bitmap on released objects");
    vm::Bitmap &bitmap = map.getBitmap();
    if (bitmap.isReleased()) {
        bitmap.acquire();
        map.forEachObject(&Object::acquireWithBitmap);
    }
#endif

    Object *obj = map.getObject(addr);
    if(obj == NULL) {
        trace::ExitCurrentFunction();
        return false;
    }
    TRACE(LOCAL,"Write access for object %p: %p", obj->addr(), addr);
    if(obj->signalWrite(addr) != gmacSuccess) ret = false;
    obj->decRef();
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t
Manager::memset(core::Mode &mode, hostptr_t s, int c, size_t size)
{
    trace::EnterCurrentFunction();
    core::Mode *owner = proc_.owner(s, size);
        if (owner == NULL) {
        ::memset(s, c, size);
        trace::ExitCurrentFunction();
        return gmacSuccess;
    }

#ifdef USE_VM
    CFATAL(owner->releasedObjects() == false, "Acquiring bitmap on released objects");
    vm::Bitmap &bitmap = owner->getBitmap();
    if (bitmap.isReleased()) {
        bitmap.acquire();
        owner->forEachObject(&Object::acquireWithBitmap);
    }
#endif

    gmacError_t ret = gmacSuccess;

    memory::ObjectMap &map = owner->getAddressSpace();
    Object *obj = map.getObject(s);
    ASSERTION(obj != NULL);
    // Check for a fast path -- probably the user is just
    // initializing a single object or a portion of an object
    if(obj->addr() <= s && obj->end() >= (s + size)) {
        size_t objSize = (size < obj->size()) ? size : obj->size();
        ret = obj->memset(s - obj->addr(), c, objSize);
        obj->decRef();
        trace::ExitCurrentFunction();
        return ret;
    }

    // This code handles the case of the user initializing a portion of
    // memory that includes host memory and GMAC objects.
    size_t left = size;
    while(left > 0) {
        // If there is no object, initialize the remaining host memory
        if(obj == NULL) {
            ::memset(s, c, left);
            left = 0; // This will finish the loop
        } else {
            // Check if there is a memory gap of host memory at the begining of the
            // memory range that remains to be initialized
            int gap = int(obj->addr() - s);
            if(gap > 0) { // If there is gap, initialize and advance the pointer
                ::memset(s, c, gap);
                left -= gap;
                s += gap;
                gap = 0;
            }
            // Check the size of the memory range from the current pointer to the end of the object
            // We add the gap, because if the ptr is within the object, its value will be negative
            size_t objSize = obj->size() + gap;
            // If the remaining memory in the object is larger than the remaining memory range, adjust
            // the size of the memory range to be initialized by the object
            objSize = (objSize < left) ? objSize : left;
            ret = obj->memset(s - obj->addr(), c, objSize);
            if(ret != gmacSuccess) break;
            left -= objSize; // Account for the bytes initialized by the object
            s += objSize;  // Advance the pointer
            obj->decRef();  // Release the object (it will not be needed anymore)
        }
        // Get the next object in the memory range that remains to be initialized
        if (left > 0) {
            memory::ObjectMap &map = owner->getAddressSpace();
            obj = map.getObject(s);
        }
    }

    trace::ExitCurrentFunction();
    return ret;
}

/**
 * Gets the number of bytes at the begining of a range that are in host memory
 * \param addr Starting address of the memory range
 * \param size Size (in bytes) of the memory range
 * \param obj First object within the range
 * \return Number of bytes at the beginning of the range that are in host memory
 */
static size_t
hostMemory(hostptr_t addr, size_t size, const Object *obj)
{
    // There is no object, so everything is in host memory
    if(obj == NULL) return size;

    // The object starts after the memory range, return the difference
    if(addr < obj->addr()) return obj->addr() - addr;

    ASSERTION(obj->end() > addr); // Sanity check
    return 0;
}

gmacError_t
Manager::memcpy(core::Mode &mode, hostptr_t dst, const hostptr_t src,
                size_t size)
{
    trace::EnterCurrentFunction();
    core::Mode *dstMode = proc_.owner(dst, size);
    core::Mode *srcMode = proc_.owner(src, size);

    if(dstMode == NULL && srcMode == NULL) {
        ::memcpy(dst, src, size);
        trace::ExitCurrentFunction();
        return gmacSuccess;
    }

    Object *dstObject = NULL;
    Object *srcObject = NULL;

    memory::ObjectMap *dstMap = NULL;
    memory::ObjectMap *srcMap = NULL;

    // Get initial objects
    if(dstMode != NULL) {
        dstMap = &dstMode->getAddressSpace();
        dstObject = dstMap->getObject(dst, size);
    }
    if(srcMode != NULL) {
        srcMap = &srcMode->getAddressSpace();
        srcObject = srcMap->getObject(src, size);
    }

    gmacError_t ret = gmacSuccess;
    size_t left = size;
    size_t offset = 0;
    size_t copySize = 0;
    while(left > 0) {
        // Get next objects involved, if necessary
        if(dstMode != NULL && dstObject != NULL && dstObject->end() < (dst + offset)) {
            dstObject->decRef();
            dstObject = dstMap->getObject(dst + offset, left);
        }
        if(srcMode != NULL && srcObject != NULL && srcObject->end() < (src + offset)) {
            srcObject->decRef();
            srcObject = srcMap->getObject(src + offset, left);
        }

        // Get the number of host-to-host memory we have to copy
        size_t dstHostMemory = hostMemory(dst + offset, left, dstObject);
        size_t srcHostMemory = hostMemory(src + offset, left, srcObject);

        if(dstHostMemory != 0 && srcHostMemory != 0) { // Host-to-host memory copy
            copySize = (dstHostMemory < srcHostMemory) ? dstHostMemory : srcHostMemory;
            ::memcpy(dst + offset, src + offset, copySize);
            ret = gmacSuccess;
        }
        else if(dstHostMemory != 0) { // Object-to-host memory copy
            size_t srcCopySize = srcObject->end() - src - offset;
            copySize = (dstHostMemory < srcCopySize) ? dstHostMemory : srcCopySize;
            size_t srcObjectOffset = src + offset - srcObject->addr();
            ret = srcObject->memcpyFromObject(mode, dst + offset, srcObjectOffset, copySize);
        }
        else if(srcHostMemory != 0) { // Host-to-object memory copy
            size_t dstCopySize = dstObject->end() - dst - offset;
            copySize = (srcHostMemory < dstCopySize) ? srcHostMemory : dstCopySize;
            size_t dstObjectOffset = dst + offset - dstObject->addr();
            ret = dstObject->memcpyToObject(mode, dstObjectOffset, src + offset, copySize);
        }
        else { // Object-to-object memory copy
            size_t srcCopySize = srcObject->end() - src - offset;
            size_t dstCopySize = dstObject->end() - dst - offset;
            copySize = (srcCopySize < dstCopySize) ? srcCopySize : dstCopySize;
            copySize = (copySize < left) ? copySize : left;
            size_t srcObjectOffset = src + offset - srcObject->addr();
            size_t dstObjectOffset = dst + offset - dstObject->addr();
            ret = srcObject->memcpyObjectToObject(mode, *dstObject, dstObjectOffset, srcObjectOffset, copySize);
        }

        offset += copySize;
        left -= copySize;
    }

    if(dstObject != NULL) dstObject->decRef();
    if(srcObject != NULL) srcObject->decRef();

    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t
Manager::flushDirty(core::Mode &mode)
{
    gmacError_t ret;
    TRACE(LOCAL,"Flushing Objects");
    // Release per-mode objects
    memory::ObjectMap &map = mode.getAddressSpace();
    ret = map.getProtocol().flushDirty();

    if(ret == gmacSuccess) {
        // Release global per-process objects
        Protocol *protocol = proc_.getProtocol();
        if(protocol != NULL) protocol->flushDirty();
    }
    return ret;
}

#if 0
gmacError_t Manager::moveTo(hostptr_t addr, core::Mode &mode)
{
    Object * obj = mode.getObjectWrite(addr);
    if(obj == NULL) return gmacErrorInvalidValue;

    mode.putObject(*obj);
#if 0
    StateObject<T>::lockWrite();
    typename StateObject<T>::SystemMap::iterator i;
    int idx = 0;
    for(i = StateObject<T>::systemMap.begin(); i != StateObject<T>::systemMap.end(); i++) {
        gmacError_t ret = accelerator->get(idx++ * paramPageSize, i->second);
    }

    owner_->free(accelerator->addr());

    StateObject<T>::unlock();
#endif
    return gmacSuccess;
}
#endif
}}
