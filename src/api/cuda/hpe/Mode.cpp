#include "api/cuda/hpe/Accelerator.h"
#include "api/cuda/hpe/Mode.h"
#include "api/cuda/hpe/Context.h"
#include "api/cuda/IOBuffer.h"

#include "util/allocator/Buddy.h"

namespace __impl { namespace cuda { namespace hpe {

Mode::Mode(core::hpe::Process &proc, Accelerator &acc, core::hpe::AddressSpace &aSpace) :
    gmac::core::hpe::Mode(proc, acc, aSpace)
#ifdef USE_VM
    //, bitmap_(*this)
#endif
{
    switchIn();
#ifdef USE_MULTI_CONTEXT
    cudaCtx_ = getAccelerator().createCUcontext();
    modules_ = getAccelerator().createModules();
#else
    modules_ = &getAccelerator().createModules();
#endif

    ModuleVector::const_iterator i;
#ifdef USE_MULTI_CONTEXT
    for(i = modules_.begin(); i != modules_.end(); ++i) {
#else
    for(i = modules_->begin(); i != modules_->end(); ++i) {
#endif
        (*i)->registerKernels(*this);
    }

    hostptr_t addr = NULL;

    gmacError_t ret = getAccelerator().hostAlloc(addr, util::params::ParamIOMemory/2, GMAC_PROT_READWRITE);
    if(ret == gmacSuccess)
        ioMemoryRead_ = new gmac::util::allocator::Buddy(addr, util::params::ParamIOMemory/2);
    ret = getAccelerator().hostAlloc(addr, util::params::ParamIOMemory/2, GMAC_PROT_READ);
    if(ret == gmacSuccess)
        ioMemoryWrite_ = new gmac::util::allocator::Buddy(addr, util::params::ParamIOMemory/2);

    streamLaunch_        = getAccelerator().createCUstream();
    streamToAccelerator_ = getAccelerator().createCUstream();
    streamToHost_        = getAccelerator().createCUstream();

    switchOut();
}

Mode::~Mode()
{
    switchIn();

    // We need to ensure that contexts are destroyed before the Mode
    cleanUpContexts();

    getAccelerator().destroyCUstream(streamLaunch_);
    getAccelerator().destroyCUstream(streamToAccelerator_);
    getAccelerator().destroyCUstream(streamToHost_);

    ModuleVector::const_iterator m;
#ifdef USE_MULTI_CONTEXT
    getAccelerator().destroyModules(modules_);
#endif
    if(ioMemoryRead_ != NULL) {
        hostFree(ioMemoryRead_->addr());
        delete ioMemoryRead_;
    }
    if(ioMemoryWrite_ != NULL) {
        hostFree(ioMemoryWrite_->addr());
        delete ioMemoryWrite_;
    }

    switchOut();
}

core::IOBuffer &
Mode::createIOBuffer(size_t size, GmacProtection prot)
{
    IOBuffer *ret;
    hostptr_t addr = NULL;
    if (prot == GMAC_PROT_WRITE) {
        if(ioMemoryWrite_ == NULL || (addr = ioMemoryWrite_->get(size)) == NULL) {
            addr = hostptr_t(::malloc(size));
            ret = new IOBuffer(addr, size, false, prot);
        } else {
            ret = new IOBuffer(addr, size, true, prot);
        }
    } else {
        if(ioMemoryRead_ == NULL || (addr = ioMemoryRead_->get(size)) == NULL) {
            addr = hostptr_t(::malloc(size));
            ret = new IOBuffer(addr, size, false, prot);
        } else {
            ret = new IOBuffer(addr, size, true, prot);
        }
    }
    return *ret;
}

void Mode::destroyIOBuffer(core::IOBuffer &buffer)
{
    ASSERTION(ioMemoryWrite_ != NULL || ioMemoryRead_ != NULL);
    if (buffer.async()) {
        if (buffer.getProtection() == GMAC_PROT_WRITE) {
            ioMemoryWrite_->put(buffer.addr(), buffer.size());
        } else {
            ioMemoryRead_->put(buffer.addr(), buffer.size());
        }
    } else {
        ::free(buffer.addr());
    }
    delete &buffer;
}

void Mode::load()
{
#ifdef USE_MULTI_CONTEXT
    cudaCtx_ = getAccelerator().createCUcontext();
#endif

#ifdef USE_MULTI_CONTEXT
    modules_ = getAccelerator().createModules();
#else
    modules_ = &getAccelerator().createModules();
#endif

    ModuleVector::const_iterator i;
#ifdef USE_MULTI_CONTEXT
    for(i = modules_.begin(); i != modules_.end(); ++i) {
#else
    for(i = modules_->begin(); i != modules_->end(); ++i) {
#endif
        (*i)->registerKernels(*this);
    }

}

void Mode::reload()
{
#ifdef USE_MULTI_CONTEXT
    getAccelerator().destroyModules(modules_);
    modules_.clear();
#endif
    kernels_.clear();
    load();
}

core::hpe::Context &Mode::getContext()
{
        core::hpe::Context *context = contextMap_.find(util::GetThreadId());
    if(context != NULL) return *context;
    context = ContextFactory::create(*this, streamLaunch_, streamToAccelerator_, streamToHost_, streamToAccelerator_);
    CFATAL(context != NULL, "Error creating new context");
        contextMap_.add(util::GetThreadId(), context);
    return *context;
}

Context &Mode::getCUDAContext()
{
    return dynamic_cast<Context &>(getContext());
}

void
Mode::destroyContext(core::hpe::Context &context) const
{
    ContextFactory::destroy(dynamic_cast<Context &>(context));
}

gmacError_t
Mode::hostAlloc(hostptr_t &addr, size_t size)
{
    switchIn();
    gmacError_t ret = getAccelerator().hostAlloc(addr, size, GMAC_PROT_READWRITE);
    switchOut();
    return ret;
}

gmacError_t Mode::hostFree(hostptr_t addr)
{
    switchIn();
    gmacError_t ret = getAccelerator().hostFree(addr);
    switchOut();
    return ret;
}

accptr_t Mode::hostMapAddr(const hostptr_t addr)
{
    switchIn();
    accptr_t ret = getAccelerator().hostMap(addr);
    switchOut();
    return ret;
}

const Variable *Mode::constant(gmacVariable_t key) const
{
    ModuleVector::const_iterator m;
#ifdef USE_MULTI_CONTEXT
    for(m = modules_.begin(); m != modules_.end(); ++m) {
#else
    for(m = modules_->begin(); m != modules_->end(); ++m) {
#endif
        const Variable *var = (*m)->constant(key);
        if(var != NULL) return var;
    }
    return NULL;
}

const Variable *Mode::variable(gmacVariable_t key) const
{
    ModuleVector::const_iterator m;
#ifdef USE_MULTI_CONTEXT
    for(m = modules_.begin(); m != modules_.end(); ++m) {
#else
    for(m = modules_->begin(); m != modules_->end(); ++m) {
#endif
        const Variable *var = (*m)->variable(key);
        if(var != NULL) return var;
    }
    return NULL;
}

const Variable *Mode::constantByName(std::string name) const
{
    ModuleVector::const_iterator m;
#ifdef USE_MULTI_CONTEXT
    for(m = modules_.begin(); m != modules_.end(); ++m) {
#else
    for(m = modules_->begin(); m != modules_->end(); ++m) {
#endif
        const Variable *var = (*m)->constantByName(name);
        if(var != NULL) return var;
    }
    return NULL;
}

const Variable *Mode::variableByName(std::string name) const
{
    ModuleVector::const_iterator m;
#ifdef USE_MULTI_CONTEXT
    for(m = modules_.begin(); m != modules_.end(); ++m) {
#else
    for(m = modules_->begin(); m != modules_->end(); ++m) {
#endif
        const Variable *var = (*m)->variableByName(name);
        if(var != NULL) return var;
    }
    return NULL;
}

const Texture *Mode::texture(gmacTexture_t key) const
{
    ModuleVector::const_iterator m;
#ifdef USE_MULTI_CONTEXT
    for(m = modules_.begin(); m != modules_.end(); ++m) {
#else
    for(m = modules_->begin(); m != modules_->end(); ++m) {
#endif
        const Texture *tex = (*m)->texture(key);
        if(tex != NULL) return tex;
    }
    return NULL;
}

gmacError_t Mode::waitForEvent(CUevent event, bool fromCUDA)
{
    // Backend methods do not need to switch in/out
    if (!fromCUDA) switchIn();
    Accelerator &acc = dynamic_cast<Accelerator &>(getAccelerator());

    CUresult ret;
    while ((ret = acc.queryCUevent(event)) == CUDA_ERROR_NOT_READY) {
        // TODO: add delay here
    }

    if (!fromCUDA) switchOut();
    return Accelerator::error(ret);
}

}}}
