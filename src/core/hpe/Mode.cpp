#include "memory/Manager.h"
#include "memory/Object.h"

#include "core/IOBuffer.h"

#include "core/hpe/Accelerator.h"
#include "core/hpe/Kernel.h"
#include "core/hpe/Mode.h"
#include "core/hpe/Context.h"
#include "core/hpe/Process.h"

namespace __impl { namespace core { namespace hpe {

Mode::Mode(Process &proc, Accelerator &acc, AddressSpace &aSpace) :
    proc_(proc),
    acc_(&acc),
    aSpace_(&aSpace),
#if defined(_MSC_VER)
#	pragma warning( disable : 4355 )
#endif
#ifdef USE_VM
    bitmap_(*this),
#endif
    contextMap_(*this)
{
}

Mode::~Mode()
{
    acc_->unregisterMode(*this);
}

void
Mode::registerKernel(gmac_kernel_id_t k, Kernel &kernel)
{
    TRACE(LOCAL,"CTX: %p Registering kernel %s: %p", this, kernel.getName(), k);
    KernelMap::iterator i;
    i = kernels_.find(k);
    ASSERTION(i == kernels_.end());
    kernels_[k] = &kernel;
}

std::string
Mode::getKernelName(gmac_kernel_id_t k) const
{
    KernelMap::const_iterator i;
    i = kernels_.find(k);
    ASSERTION(i != kernels_.end());
    return std::string(i->second->getName());
}

gmacError_t
Mode::map(accptr_t &dst, hostptr_t src, size_t count, unsigned align)
{
    switchIn();

    gmacError_t ret;
    accptr_t acc(0);
    bool hasMapping = false;
    if (src != NULL) {
        hasMapping = acc_->getMapping(acc, src, count);
    }
    if (hasMapping == true) {
        ret = gmacSuccess;
        dst = acc;
        TRACE(LOCAL,"Mapping for address %p: %p", src, dst.get());
    } else {
        ret = acc_->map(dst, src, count, align);
        TRACE(LOCAL,"New Mapping for address %p: %p", src, dst.get());
    }

#ifdef USE_MULTI_CONTEXT
    dst.pasId_ = this->getId();
#endif

    switchOut();
    return ret;
}

gmacError_t
Mode::add_mapping(accptr_t dst, hostptr_t src, size_t count)
{
    switchIn();
    gmacError_t ret = acc_->add_mapping(dst, src, count);
    switchOut();

    return ret;
}

gmacError_t
Mode::unmap(hostptr_t addr, size_t count)
{
    switchIn();
    gmacError_t ret = acc_->unmap(addr, count);
    switchOut();
    return ret;
}

gmacError_t
Mode::copyToAccelerator(accptr_t acc, const hostptr_t host, size_t count)
{
    TRACE(LOCAL,"Copy %p to accelerator %p ("FMT_SIZE" bytes)", host, acc.get(), count);

    switchIn();
    gmacError_t ret = getContext().copyToAccelerator(acc, host, count);
    switchOut();

    return ret;
}

gmacError_t
Mode::copyToHost(hostptr_t host, const accptr_t acc, size_t count)
{
    TRACE(LOCAL,"Copy %p to host %p ("FMT_SIZE" bytes)", acc.get(), host, count);

    switchIn();
    gmacError_t ret = getContext().copyToHost(host, acc, count);
    switchOut();

    return ret;
}

gmacError_t
Mode::copyAccelerator(accptr_t dst, const accptr_t src, size_t count)
{
    switchIn();
    gmacError_t ret = acc_->copyAccelerator(dst, src, count, streamToHost_);
    switchOut();
    return ret;
}

gmacError_t
Mode::memset(accptr_t addr, int c, size_t count)
{
    switchIn();
    gmacError_t ret = acc_->memset(addr, c, count, streamLaunch_);
    switchOut();
    return ret;
}

// Nobody can enter GMAC until this has finished. No locks are needed
gmacError_t
Mode::moveTo(Accelerator &acc)
{
    TRACE(LOCAL,"Moving mode from acc %d to %d", acc_->id(), acc.id());
    switchIn();

    if (acc_ == &acc) {
        switchOut();
        return gmacSuccess;
    }
    gmacError_t ret = gmacSuccess;
    size_t free;
    size_t total;
    size_t needed = aSpace_->memorySize();
    acc_->getMemInfo(free, total);

    if (needed > free) {
        switchOut();
        return gmacErrorInsufficientAcceleratorMemory;
    }

    TRACE(LOCAL,"Releasing object memory in accelerator");
    ret = aSpace_->forEachObject(&memory::Object::unmapFromAccelerator);

    TRACE(LOCAL,"Cleaning contexts");
    contextMap_.clean();

    TRACE(LOCAL,"Registering mode in new accelerator");
    acc_->migrateMode(*this, acc);

    TRACE(LOCAL,"Reallocating objects");
    // aSpace_->reallocObjects(*this);
    ret = aSpace_->forEachObject(&memory::Object::mapToAccelerator);

    TRACE(LOCAL,"Reloading mode");
    reload();

    switchOut();

    return ret;
}

gmacError_t Mode::cleanUp()
{
    gmacError_t ret = aSpace_->forEachObject<core::Mode>(&memory::Object::removeOwner, *this);
    contextMap_.clean();
    aSpace_->cleanUp();
#ifdef USE_VM
    bitmap_.cleanUp();
#endif
    return ret;
}

void
Mode::setAddressSpace(AddressSpace &aSpace)
{
    aSpace_ = &aSpace;
}

}}}
