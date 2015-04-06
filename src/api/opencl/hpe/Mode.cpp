#include "api/opencl/IOBuffer.h"

#include "api/opencl/hpe/Accelerator.h"
#include "api/opencl/hpe/Context.h"
#include "api/opencl/hpe/Mode.h"

namespace __impl { namespace opencl { namespace hpe {

Mode::Mode(core::hpe::Process &proc, Accelerator &acc, core::hpe::AddressSpace &aSpace) :
    gmac::core::hpe::Mode(proc, acc, aSpace)
{
    streamLaunch_ = getAccelerator().createCLstream();
    streamToHost_ = streamToAccelerator_ = streamLaunch_;
}

Mode::~Mode()
{
    // We need to ensure that contexts are destroyed before the Mode
    getAccelerator().destroyCLstream(streamLaunch_);
}

core::IOBuffer &Mode::createIOBuffer(size_t size, GmacProtection prot)
{
    IOBuffer *ret;
    hostptr_t addr(NULL);
    cl_mem mem;
    gmacError_t err = getAccelerator().allocCLBuffer(mem, addr, size, prot);
    if(err != gmacSuccess) {
        addr = hostptr_t(::malloc(size));
        ret = new IOBuffer(*this, addr, size, NULL, prot);
    } else {
        ret = new IOBuffer(*this, addr, size, mem, prot);
    }
    return *ret;
}

void Mode::destroyIOBuffer(core::IOBuffer &_buffer)
{
    IOBuffer &buffer = dynamic_cast<IOBuffer &>(_buffer);
    if (buffer.async()) {
        getAccelerator().freeCLBuffer(buffer.getCLBuffer(), buffer.addr(), buffer.size(), buffer.getProtection());
    } else {
        ::free(buffer.addr());
    }
    delete &buffer;
}


void Mode::reload()
{
}

core::hpe::Context &Mode::getContext()
{
    core::hpe::Context *context = contextMap_.find(util::GetThreadId());
    if(context != NULL) return *context;
    context = ContextFactory::create(*this, streamLaunch_);
    CFATAL(context != NULL, "Error creating new context");
    contextMap_.add(util::GetThreadId(), context);
    return *context;
}

Context &Mode::getCLContext()
{
    return dynamic_cast<Context &>(getContext());
}

void Mode::destroyContext(core::hpe::Context &context) const
{
    ContextFactory::destroy(dynamic_cast<Context &>(context));
}

gmacError_t Mode::hostAlloc(hostptr_t &addr, size_t size)
{
    switchIn();
    gmacError_t ret = getAccelerator().hostAlloc(addr, size);
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
    accptr_t ret = getAccelerator().hostMapAddr(addr);
    switchOut();
    return ret;
}

gmacError_t Mode::launch(gmac_kernel_id_t name, core::hpe::KernelLaunch *&launch)
{
    KernelMap::iterator i = kernels_.find(name);
    Kernel *k = NULL;
    if (i == kernels_.end()) {
        k = dynamic_cast<Accelerator *>(acc_)->getKernel(name);
        if(k == NULL) return gmacErrorInvalidValue;
        registerKernel(name, *k);
        kernelList_.insert(k);
    }
    else k = dynamic_cast<Kernel *>(i->second);
    switchIn();
    launch = &(getCLContext().launch(*k));
    switchOut();
    return gmacSuccess;
}


gmacError_t Mode::execute(core::hpe::KernelLaunch & launch)
{
    switchIn();
    gmacError_t ret = proc_.prepareForCall();

    if(ret == gmacSuccess) {
        ret = getAccelerator().execute(dynamic_cast<KernelLaunch &>(launch));
    }
    switchOut();
    return ret;
}

gmacError_t Mode::waitForEvent(cl_event event)
{
    switchIn();
    Accelerator &acc = dynamic_cast<Accelerator &>(getAccelerator());

    gmacError_t ret = acc.syncCLevent(event);
    switchOut();
    return ret;

    // TODO: try to implement wait as a polling loop -- AMD OpenCL blocks execution
#if 0
    cl_int ret;
    while ((ret = acc.queryCLevent(event)) != CL_COMPLETE) {
        // TODO: add delay here
    }

        switchOut();

    return Accelerator::error(ret);
#endif
}

Accelerator & Mode::getAccelerator() const
{
    return *static_cast<Accelerator *>(acc_);
}

gmacError_t
Mode::acquire(hostptr_t addr)
{
    switchIn();
    Accelerator &acc = dynamic_cast<Accelerator &>(getAccelerator());

    gmacError_t ret = acc.acquire(addr);
    switchOut();
    return ret;
}

gmacError_t
Mode::release(hostptr_t addr)
{
    switchIn();
    Accelerator &acc = dynamic_cast<Accelerator &>(getAccelerator());

    gmacError_t ret = acc.release(addr);
    switchOut();
    return ret;
}



}}}
