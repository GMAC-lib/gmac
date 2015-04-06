
#include "api/opencl/hpe/Mode.h"

#include "api/opencl/IOBuffer.h"
#include "api/opencl/Tracer.h"

#include "core/Process.h"

#include "hpe/init.h"

#if !defined(_MSC_VER) && !defined(__APPLE__)
// Symbols needed for automatic compilation of embedded code
extern "C" {
    extern char __ocl_code_start __attribute__((weak));
    extern char __ocl_code_end   __attribute__((weak));
};
#endif

namespace __impl { namespace opencl { namespace hpe {

void
CLBufferPool::cleanUp(stream_t stream)
{
    CLMemMap::iterator it;

    for(it = begin(); it != end(); ++it) {
        CLMemList::iterator jt;
        CLMemList list = it->second;
        for (jt = list.begin(); jt != list.end(); ++jt) {
            cl_int ret;
            ret = clEnqueueUnmapMemObject(stream, jt->first, jt->second, 0, NULL, NULL);
            ASSERTION(ret == CL_SUCCESS);
            ret = clReleaseMemObject(jt->first);
            ASSERTION(ret == CL_SUCCESS);
        }
    }

}

bool
CLBufferPool::getCLMem(size_t size, cl_mem &mem, hostptr_t &addr)
{
    lock();

    bool ret = false;

    CLMemMap::iterator it = find(size);

    if (it != end())  {
        CLMemList &list = it->second;
        if (it->second.size() > 0) {
            mem = list.front().first;
            addr = list.front().second;
            list.pop_front();
            ret = true;
        }
    }
    unlock();

    return ret;
}

void CLBufferPool::putCLMem(size_t size, cl_mem mem, hostptr_t addr)
{
    lock();

    CLMemMap::iterator it = find(size);

    if (it != end())  {
        CLMemList &list = it->second;
        list.push_back(std::make_pair(mem, addr));
    } else {
        CLMemList list;
        list.push_back(std::make_pair(mem, addr));
        insert(CLMemMap::value_type(size, list));
    }
    unlock();
}

Accelerator::AcceleratorMap *Accelerator::Accelerators_ = NULL;
HostMap *Accelerator::GlobalHostAlloc_ = NULL;

Accelerator::Accelerator(int n, cl_context context, cl_device_id device, unsigned major, unsigned minor) :
    gmac::util::SpinLock("Accelerator"),
    gmac::core::hpe::Accelerator(n),
    ctx_(context), device_(device),
    major_(major), minor_(minor),
    allocatedMemory_(0),
    isInfoInitialized_(false),
    acceleratorName_(NULL),
    vendorName_(NULL),
    maxSizes_(NULL)
{
    // Not used for now
    busId_ = 0;
    busAccId_ = 0;

#if defined(__APPLE__)
    integrated_ = false;
    cl_int ret = CL_SUCCESS;
#else 
    cl_bool val = CL_FALSE;
    cl_int ret = clGetDeviceInfo(device_, CL_DEVICE_HOST_UNIFIED_MEMORY,
        sizeof(val), NULL, NULL);
    if(ret == CL_SUCCESS) integrated_ = (val == CL_TRUE);
    else integrated_ = false;
#endif

    TRACE(LOCAL, "Created OpenCL accelerator with capability %u.%u", major_, minor_);

    ret = clRetainContext(ctx_);
    CFATAL(ret == CL_SUCCESS, "Unable to retain OpenCL context");
}

Accelerator::~Accelerator()
{
    std::vector<cl_program> &programs = (*Accelerators_)[this];
    std::vector<cl_program>::const_iterator i;
    // We cannot call OpenCL at destruction time because the library
    // might have been unloaded
    lock();
    for(i = programs.begin(); i != programs.end(); ++i) {
        cl_int ret = clReleaseProgram(*i);
        ASSERTION(ret == CL_SUCCESS);
    }
    unlock();
    stream_t tmpStream = createCLstream();
    clMemWrite_.cleanUp(tmpStream);
    clMemRead_.cleanUp(tmpStream);
    destroyCLstream(tmpStream);
    Accelerators_->erase(this);

    if (Accelerators_->size() == 0) {
        delete Accelerators_;
        delete GlobalHostAlloc_;
        Accelerators_ = NULL;
    }

    if (acceleratorName_ != NULL) delete [] acceleratorName_;
    if (vendorName_ != NULL) delete [] vendorName_;
    if (maxSizes_ != NULL) delete [] maxSizes_;

    cl_int ret = CL_SUCCESS;
    ret = clReleaseContext(ctx_);
    ASSERTION(ret == CL_SUCCESS);
}

void Accelerator::init()
{
}

core::hpe::Mode *Accelerator::createMode(core::hpe::Process &proc, core::hpe::AddressSpace &aSpace)
{
    trace::EnterCurrentFunction();
    core::hpe::Mode *mode = ModeFactory::create(dynamic_cast<core::hpe::Process &>(proc), *this, aSpace);
    if (mode != NULL) {
        registerMode(*mode);
    }
    trace::ExitCurrentFunction();

    TRACE(LOCAL, "Creating Execution Mode %p to Accelerator", mode);
    return mode;
}

gmacError_t Accelerator::map(accptr_t &dst, hostptr_t src, size_t size, unsigned /*align*/)
{
    trace::EnterCurrentFunction();

    cl_int ret = CL_SUCCESS;
    trace::SetThreadState(trace::Wait);
    dst(clCreateBuffer(ctx_, CL_MEM_READ_WRITE, size, NULL, &ret));
    if(ret != CL_SUCCESS) return error(ret);
    trace::SetThreadState(trace::Running);
    allocatedMemory_ += size;

    dst.pasId_ = id_;

    TRACE(LOCAL, "Allocating accelerator memory (%d bytes) @ %p", size, dst.get());

    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t
Accelerator::add_mapping(accptr_t dst, hostptr_t src, size_t size)
{
    allocations_.insert(src, dst, size);

    return gmacSuccess;
}

gmacError_t Accelerator::unmap(hostptr_t host, size_t size)
{
    trace::EnterCurrentFunction();
    ASSERTION(host != NULL);

    accptr_t addr;
    size_t s;

    bool hasMapping = allocations_.find(host, addr, s);
    ASSERTION(hasMapping == true);
    ASSERTION(s == size);
    allocations_.erase(host, size);

    TRACE(LOCAL, "Releasing accelerator memory @ %p", addr.get());

    trace::SetThreadState(trace::Wait);
    cl_int ret = CL_SUCCESS;
    ret = clReleaseMemObject(addr.get());
    allocatedMemory_ -= size;
    trace::SetThreadState(trace::Running);
    trace::ExitCurrentFunction();
    return error(ret);
}


gmacError_t Accelerator::copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size, core::hpe::Mode &mode)
{
    trace::EnterCurrentFunction();
    if (cmd_.empty() == true) createCLstream();
    TRACE(LOCAL, "Copy to accelerator: %p ("FMT_SIZE") @ %p", host, size, acc.get());
    trace::SetThreadState(trace::Wait);
    cl_event event;
    trace_.init(trace_.getThreadId(), (THREAD_T)mode.getId());
    lock();
    cl_int ret = clEnqueueWriteBuffer(cmd_.front(), acc.get(),
        CL_TRUE, acc.offset(), size, host, 0, NULL, &event);
    unlock();
    CFATAL(ret == CL_SUCCESS, "Error copying to accelerator: %d", ret);
    trace_.trace(event, event, size);
    trace::SetThreadState(trace::Running);
    cl_int clret = clReleaseEvent(event);
    ASSERTION(clret == CL_SUCCESS);
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::copyToHost(hostptr_t host, const accptr_t acc, size_t count, core::hpe::Mode &mode)
{
    trace::EnterCurrentFunction();
    if (cmd_.empty() == true) createCLstream();
    TRACE(LOCAL, "Copy to host: %p ("FMT_SIZE") @ %p", host, count, acc.get());
    trace::SetThreadState(trace::Wait);
    cl_event event;
    trace_.init((THREAD_T)mode.getId(), trace_.getThreadId());
    lock();
    cl_int ret = clEnqueueReadBuffer(cmd_.front(), acc.get(),
        CL_TRUE, acc.offset(), count, host, 0, NULL, &event);
    unlock();
    CFATAL(ret == CL_SUCCESS, "Error copying to host: %d", ret);
    trace_.trace(event, event, count);
    trace::SetThreadState(trace::Running);
    cl_int clret = clReleaseEvent(event);
    ASSERTION(clret == CL_SUCCESS);
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::copyAccelerator(accptr_t dst, const accptr_t src, size_t size, stream_t stream)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL, "Copy accelerator-accelerator ("FMT_SIZE") @ %p:"FMT_SIZE" - %p:"FMT_SIZE, size,
          src.get(), src.offset(),
          dst.get(), src.offset());
    lock();
    cl_event event;
    cl_int ret = clEnqueueCopyBuffer(stream, src.get(), dst.get(), src.offset(), dst.offset(), size, 0, NULL, &event);
    ASSERTION(ret == CL_SUCCESS);
    ret = clWaitForEvents(1, &event);
    ASSERTION(ret == CL_SUCCESS);
    unlock();
    trace::ExitCurrentFunction();
    return error(ret);
}


gmacError_t Accelerator::memset(accptr_t addr, int c, size_t size, stream_t stream)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL, "Setting accelerator memory ("FMT_SIZE") @ %p", size, addr.get());
    // TODO: This is a very inefficient implementation. We might consider
    // using a kernel for this task
    void *tmp = ::malloc(size);
    ::memset(tmp, c, size);
    lock();
    cl_int ret = clEnqueueWriteBuffer(stream, addr.get(),
        CL_TRUE, addr.offset(), size, tmp, 0, NULL, NULL);
    unlock();
    ::free(tmp);
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::sync()
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL, "Waiting for accelerator to finish all activities");
    cl_int ret = cmd_.sync();
    trace::ExitCurrentFunction();
    return error(ret);
}

void
Accelerator::addAccelerator(Accelerator &acc)
{
    std::pair<Accelerator *, std::vector<cl_program> > pair(&acc, std::vector<cl_program>());
    if (Accelerators_ == NULL) {
        Accelerators_ = new AcceleratorMap();
        GlobalHostAlloc_ = new HostMap();
    }
    Accelerators_->insert(pair);
}

Kernel *
Accelerator::getKernel(gmac_kernel_id_t k)
{
    std::vector<cl_program> &programs = (*Accelerators_)[this];
    ASSERTION(programs.size() > 0);

    TRACE(LOCAL, "Creating kernels");
    cl_int err = CL_SUCCESS;
    std::vector<cl_program>::const_iterator i;
    for(i = programs.begin(); i != programs.end(); ++i) {
        lock();
        cl_kernel kernel = clCreateKernel(*i, k, &err);
        unlock();
        if(err != CL_SUCCESS) continue;
        return new Kernel(__impl::core::hpe::KernelDescriptor(k, k), kernel);
    }
    return NULL;
}

gmacError_t Accelerator::prepareEmbeddedCLCode()
{
#if !defined(_MSC_VER) && !defined(__APPLE__)
    const char *CL_MAGIC = "!@#~";

    trace::EnterCurrentFunction();

    TRACE(GLOBAL, "Preparing embedded code");

    if (&__ocl_code_start != NULL &&
        &__ocl_code_end   != NULL) {
        char *code = &__ocl_code_start;
        char *cursor = strstr(code, CL_MAGIC);

        while (cursor != NULL && cursor < &__ocl_code_end) {
            char *params = cursor + strlen(CL_MAGIC);

            size_t fileSize = cursor - code;
            char *file = new char[fileSize + 1];
            char *fileParams = NULL;
            ::memcpy(&file[0], code, fileSize);
            file[fileSize] = '\0';

            cursor = ::strstr(cursor + 1, CL_MAGIC);
            if (cursor != params) {
                size_t paramsSize = cursor - params;
                fileParams = new char[paramsSize + 1];
                ::memcpy(fileParams, params, paramsSize);
                fileParams[paramsSize] = '\0';
            }
            TRACE(GLOBAL, "Compiling file in embedded code");
            gmacError_t ret = prepareCLCode(file, fileParams);
            if (ret != gmacSuccess) {
                abort();
                trace::ExitCurrentFunction();
                return error(ret);
            }

            delete [] file;
            if (fileParams != NULL) delete [] fileParams;

            code = cursor + strlen(CL_MAGIC);
            cursor = ::strstr(cursor + 1, CL_MAGIC);
        }
    }
    trace::ExitCurrentFunction();
#endif
    return gmacSuccess;
}

gmacError_t Accelerator::prepareCLCode(const char *code, const char *flags)
{
    trace::EnterCurrentFunction();

    cl_int ret;
    AcceleratorMap::iterator it;
    for (it = Accelerators_->begin(); it != Accelerators_->end(); ++it) {
        cl_program program = clCreateProgramWithSource(
            it->first->ctx_, 1, &code, NULL, &ret);
        if (ret == CL_SUCCESS) {
            ret = clBuildProgram(program, 1, &it->first->device_, flags, NULL, NULL);
        }
        if (ret == CL_SUCCESS) {
            it->second.push_back(program);
            TRACE(GLOBAL, "Compilation OK for accelerator: %d", it->first->device_);
        } else {
            size_t len;
            cl_int tmp = clGetProgramBuildInfo(program, it->first->device_,
                    CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
            ASSERTION(tmp == CL_SUCCESS);
            char *msg = new char[len + 1];
            tmp = clGetProgramBuildInfo(program, it->first->device_,
                    CL_PROGRAM_BUILD_LOG, len, msg, NULL);
            ASSERTION(tmp == CL_SUCCESS);
            msg[len] = '\0';
            TRACE(GLOBAL, "Error compiling code accelerator: %d\n%s",
                it->first->device_, msg);
            delete [] msg;
            break;
        }
    }
#ifdef USE_DEPRECATED_OPENCL_1_1
    clUnloadCompiler();
#endif
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::prepareCLBinary(const unsigned char *binary, size_t size, const char *flags)
{
    trace::EnterCurrentFunction();

    cl_int ret;
    AcceleratorMap::iterator it;
    for (it = Accelerators_->begin(); it != Accelerators_->end(); ++it) {
        cl_program program = clCreateProgramWithBinary(it->first->ctx_, 1, &it->first->device_, &size, &binary, NULL, &ret);
        if (ret == CL_SUCCESS) {
            // TODO use the callback parameter to allow background code compilation
            ret = clBuildProgram(program, 1, &it->first->device_, flags, NULL, NULL);
        }
        if (ret == CL_SUCCESS) {
            it->second.push_back(program);
            TRACE(GLOBAL, "Compilation OK for accelerator: %d", it->first->device_);
        } else {
            size_t len;
            cl_int tmp = clGetProgramBuildInfo(program, it->first->device_, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
            ASSERTION(tmp == CL_SUCCESS);
            char *msg = new char[len + 1];
            tmp = clGetProgramBuildInfo(program, it->first->device_, CL_PROGRAM_BUILD_LOG, len, msg, NULL);
            ASSERTION(tmp == CL_SUCCESS);
            msg[len] = '\0';
            TRACE(GLOBAL, "Error compiling code on accelerator %d\n%s", it->first->device_, msg);
            delete [] msg;
            break;
        }
    }
    trace::ExitCurrentFunction();
    return error(ret);
}

cl_command_queue Accelerator::createCLstream()
{
    trace::EnterCurrentFunction();
    cl_command_queue stream;
    cl_int error;
    cl_command_queue_properties prop = 0;
#if defined(USE_TRACE)
    prop |= CL_QUEUE_PROFILING_ENABLE;
#endif
    stream = clCreateCommandQueue(ctx_, device_, prop, &error);
    CFATAL(error == CL_SUCCESS, "Unable to create OpenCL stream");
    TRACE(LOCAL, "Created OpenCL stream %p, in Accelerator %p", stream, this);
    cmd_.add(stream);
    trace::ExitCurrentFunction();
    return stream;
}

void Accelerator::destroyCLstream(cl_command_queue stream)
{
    trace::EnterCurrentFunction();
        // We cannot remove the command queue because the OpenCL DLL might
        // have been already unloaded
    cl_int ret = CL_SUCCESS;
    ret = clReleaseCommandQueue(stream);
    CFATAL(ret == CL_SUCCESS, "Unable to destroy OpenCL stream");

    TRACE(LOCAL, "Destroyed OpenCL stream %p, in Accelerator %p", stream, this);
    cmd_.remove(stream);
    trace::ExitCurrentFunction();
}


gmacError_t Accelerator::syncStream(stream_t stream)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL, "Waiting for stream %p in Accelerator %p", stream, this);
    CFATAL(clFlush(stream) == CL_SUCCESS);
    cl_int ret = clFinish(stream);
    CFATAL(ret == CL_SUCCESS, "Error syncing cl_command_queue: %d", ret);
    trace::ExitCurrentFunction();
    return error(ret);
}

cl_int Accelerator::queryCLevent(cl_event event)
{
    cl_int ret = CL_SUCCESS;
    trace::EnterCurrentFunction();
    cl_int err = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
        sizeof(cl_int), &ret, NULL);
    CFATAL(err == CL_SUCCESS, "Error querying cl_event: %d", err);
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Accelerator::syncCLevent(cl_event event)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL, "Accelerator waiting for event");
#if defined(OPENCL_1_1)
        cl_int status = 0;
        cl_int ret = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, NULL);
        while(ret == CL_SUCCESS && status > 0) {
                ret = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, NULL);
        }
#else
        cl_int ret = clWaitForEvents(1, &event);
#endif
    CFATAL(ret == CL_SUCCESS, "Error syncing cl_event: %d", ret);
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::timeCLevents(uint64_t &t, cl_event start, cl_event end)
{
    uint64_t startTime, endTime;
    t = 0;
    cl_int ret = clGetEventProfilingInfo(start, CL_PROFILING_COMMAND_START,
        sizeof(startTime), &startTime, NULL);
    if(ret != CL_SUCCESS) return error(CL_SUCCESS);
    ret = clGetEventProfilingInfo(end, CL_PROFILING_COMMAND_END,
        sizeof(endTime), &endTime, NULL);
    if(ret != CL_SUCCESS) return error(CL_SUCCESS);
    t = (endTime - startTime) / 1000;
    return error(ret);
}


gmacError_t
Accelerator::hostAlloc(hostptr_t &addr, size_t size)
{
    // There is not reliable way to get zero-copy memory
    cl_int ret = CL_SUCCESS;
    cl_int flags;
    if (addr != NIL) {
        flags = CL_MEM_USE_HOST_PTR;
    } else {
        flags = CL_MEM_ALLOC_HOST_PTR;
    }

    cl_mem mem = clCreateBuffer(ctx_, flags, size, addr, &ret);
    if (ret == CL_SUCCESS) {
        stream_t stream = cmd_.front();
        // Get the host pointer for the memory object
        flags = CL_MAP_WRITE | CL_MAP_READ;
        lock();
        addr = (hostptr_t)clEnqueueMapBuffer(stream, mem, CL_TRUE,
                                             flags, 0, size, 0, NULL, NULL, &ret);
        unlock();

        // Insert the object in the allocation map for the accelerator
        if(ret != CL_SUCCESS) clReleaseMemObject(mem);
        else {
            allocatedMemory_ += size;
            localHostAlloc_.insert(addr, mem, size);
        }
    }

    return error(ret);
}

gmacError_t
Accelerator::allocCLBuffer(cl_mem &mem, hostptr_t &addr, size_t size, GmacProtection prot)
{
    trace::EnterCurrentFunction();
    cl_int ret = CL_SUCCESS;
    cl_int flags = CL_MEM_ALLOC_HOST_PTR;

    if (prot == GMAC_PROT_WRITE) {
        if (clMemRead_.getCLMem(size, mem, addr)) {
            goto exit;
        }
        flags |= CL_MEM_READ_ONLY;
    } else {
        if (clMemWrite_.getCLMem(size, mem, addr)) {
            goto exit;
        }
        flags |= CL_MEM_WRITE_ONLY;
    }
    ASSERTION(cmd_.empty() == false);

    // Get a memory object in the host memory
    mem = clCreateBuffer(ctx_, flags, size, NULL, &ret);
    if(ret == CL_SUCCESS) {
        stream_t stream = cmd_.front();
        // Get the host pointer for the memory object
        if (prot == GMAC_PROT_WRITE) {
            flags = CL_MAP_WRITE;
        } else {
            flags = CL_MAP_WRITE | CL_MAP_READ;
        }
        lock();
        addr = (hostptr_t)clEnqueueMapBuffer(stream, mem, CL_TRUE,
                                             flags, 0, size, 0, NULL, NULL, &ret);
        unlock();

        // Insert the object in the allocation map for the accelerator
        if(ret != CL_SUCCESS) clReleaseMemObject(mem);
        else allocatedMemory_ += size;
    }

exit:
    trace::ExitCurrentFunction();
    return error(ret);
}


gmacError_t Accelerator::hostFree(hostptr_t addr)
{
    trace::EnterCurrentFunction();
    // There is not reliable way to get zero-copy memory
    size_t dummy;
    cl_mem mem;
    ASSERTION(localHostAlloc_.translate(addr, mem, dummy) == true);
    localHostAlloc_.remove(addr);

    trace::ExitCurrentFunction();

    return gmacSuccess;
}

gmacError_t Accelerator::freeCLBuffer(cl_mem mem, hostptr_t addr, size_t size, GmacProtection prot)
{
    trace::EnterCurrentFunction();
    if (prot == GMAC_PROT_WRITE) {
        // clMemRead refers to the buffers readable in accelerator, written by host
        clMemRead_.putCLMem(size, mem, addr);
    } else {
        // clMemWrite refers to the buffers writable in accelerator, readable by host
        clMemWrite_.putCLMem(size, mem, addr);
    }
    trace::ExitCurrentFunction();

    return error(CL_SUCCESS);
}

accptr_t Accelerator::hostMapAddr(const hostptr_t addr)
{
    trace::EnterCurrentFunction();
    // There is not reliable way to get zero-copy memory
    size_t dummy;
    cl_mem mem;
    bool found = localHostAlloc_.translate(addr, mem, dummy);
    accptr_t acc(0);
    if (found) {
        acc = accptr_t(mem);
        acc.pasId_ = id_;
    }

    trace::ExitCurrentFunction();

    return acc;
}

gmacError_t Accelerator::execute(cl_command_queue stream, cl_kernel kernel, cl_uint workDim,
        const size_t *offset, const size_t *globalSize, const size_t *localSize, cl_event *event)
{
    TRACE(LOCAL, "Executing kernel %p", kernel);
    lock();
    cl_int ret = clEnqueueNDRangeKernel(stream, kernel, workDim, offset, globalSize, localSize,
             0, NULL, event);
        clFlush(stream);
    unlock();
    return error(ret);
}

void Accelerator::getMemInfo(size_t &free, size_t &total) const
{
    cl_int ret = CL_SUCCESS;
    cl_ulong value = 0;
    ret = clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE,
        sizeof(value), &value, NULL);
    CFATAL(ret == CL_SUCCESS , "Unable to get attribute %d", ret);
    total = size_t(value);
    free = total - allocatedMemory_;
}

void Accelerator::getAcceleratorInfo(GmacAcceleratorInfo &info)
{
    if (isInfoInitialized_ == false) {
        cl_device_type deviceType;
        cl_uint deviceVendor;

        size_t nameSize;
        cl_int res = clGetDeviceInfo(device_, CL_DEVICE_NAME, 0, NULL, &nameSize);
        ASSERTION(res == CL_SUCCESS);
        acceleratorName_ = new char[nameSize + 1];
        res = clGetDeviceInfo(device_, CL_DEVICE_NAME, nameSize, acceleratorName_, NULL);
        ASSERTION(res == CL_SUCCESS);
        acceleratorName_[nameSize] = '\0';
        res = clGetDeviceInfo(device_, CL_DEVICE_VENDOR, 0, NULL, &nameSize);
        ASSERTION(res == CL_SUCCESS);
        vendorName_ = new char[nameSize + 1];
        res = clGetDeviceInfo(device_, CL_DEVICE_VENDOR, nameSize, vendorName_, NULL);
        ASSERTION(res == CL_SUCCESS);
        vendorName_[nameSize] = '\0';

        accInfo_.acceleratorName = acceleratorName_;
        accInfo_.vendorName = vendorName_;

        res = clGetDeviceInfo(device_, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, NULL);
        ASSERTION(res == CL_SUCCESS);

        res = clGetDeviceInfo(device_, CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &deviceVendor, NULL);
        ASSERTION(res == CL_SUCCESS);

        accInfo_.acceleratorType = GmacAcceleratorType(0);
        if (deviceType & CL_DEVICE_TYPE_CPU) {
            accInfo_.acceleratorType = GmacAcceleratorType(accInfo_.acceleratorType | GMAC_ACCELERATOR_TYPE_CPU);
        }

        if (deviceType & CL_DEVICE_TYPE_GPU) {
            accInfo_.acceleratorType = GmacAcceleratorType(accInfo_.acceleratorType | GMAC_ACCELERATOR_TYPE_GPU);
        }

        if (deviceType & CL_DEVICE_TYPE_ACCELERATOR) {
            accInfo_.acceleratorType = GmacAcceleratorType(accInfo_.acceleratorType | GMAC_ACCELERATOR_TYPE_ACCELERATOR);
        }

        accInfo_.vendorId = deviceVendor;
        accInfo_.isAvailable = 1;

        cl_uint computeUnits;
        cl_uint dimensions;
        size_t workGroupSize;
        cl_ulong globalMemSize;
        cl_ulong localMemSize;
        cl_ulong cacheMemSize;

        res = clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &computeUnits, NULL);
        ASSERTION(res == CL_SUCCESS);
        res = clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &dimensions, NULL);
        ASSERTION(res == CL_SUCCESS);

        accInfo_.computeUnits = computeUnits;
        accInfo_.maxDimensions = dimensions;
        maxSizes_ = new size_t[dimensions];

        res = clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * dimensions, maxSizes_, NULL);
        ASSERTION(res == CL_SUCCESS);

        accInfo_.maxSizes = maxSizes_;

        res = clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &workGroupSize, NULL);
        ASSERTION(res == CL_SUCCESS);
        res = clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &globalMemSize, NULL);
        ASSERTION(res == CL_SUCCESS);
        res = clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemSize, NULL);
        ASSERTION(res == CL_SUCCESS);
        res = clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &cacheMemSize, NULL);
        ASSERTION(res == CL_SUCCESS);
        accInfo_.maxWorkGroupSize = workGroupSize;
        accInfo_.globalMemSize = static_cast<size_t>(globalMemSize);
        accInfo_.localMemSize  = static_cast<size_t>(localMemSize);
        accInfo_.cacheMemSize  = static_cast<size_t>(cacheMemSize);

        size_t driverSize = 0;
        res = clGetDeviceInfo(device_, CL_DRIVER_VERSION, 0, NULL, &driverSize);
        ASSERTION(res == CL_SUCCESS);
        if(driverSize > 0) {
            char *driverName = new char[driverSize + 1];
            res = clGetDeviceInfo(device_, CL_DRIVER_VERSION, driverSize, driverName, NULL);
            ASSERTION(res == CL_SUCCESS);
            std::string driverString(driverName);
            size_t number = driverString.find_first_of("1234567890");
            size_t first_dot = driverString.find_first_of('.');
            size_t last_dot = driverString.find_last_of('.');
            if(last_dot == first_dot) last_dot = driverString.length() + 1;
            if(first_dot != std::string::npos) {
                std::string majorString = driverString.substr(number, first_dot);
                accInfo_.driverMajor = atoi(majorString.c_str());
                std::string minorString = driverString.substr(first_dot + 1, last_dot);
                accInfo_.driverMinor = atoi(minorString.c_str());
                if(last_dot < driverString.length()) {
                    std::string revString = driverString.substr(last_dot + 1);
                    accInfo_.driverRev = atoi(revString.c_str());
                }
                else accInfo_.driverRev = 0;
            }
            delete [] driverName;
        }

        isInfoInitialized_ = true;
    }

    info = accInfo_;
}

gmacError_t
Accelerator::acquire(hostptr_t addr)
{
    trace::EnterCurrentFunction();
    size_t size = 0;
    cl_mem mem;
    bool found = localHostAlloc_.translate(addr, mem, size);
    ASSERTION(found == true);

    cl_int ret;
    cl_int flags = CL_MAP_WRITE | CL_MAP_READ;
    stream_t stream = cmd_.front();
    hostptr_t new_addr = (hostptr_t)clEnqueueMapBuffer(stream, mem, CL_TRUE,
                                                       flags, 0, size, 0, NULL, NULL, &ret);
    ASSERTION(addr == new_addr, "Addresses do not match");
    trace::ExitCurrentFunction();

    return error(ret);
}

gmacError_t
Accelerator::release(hostptr_t addr)
{
    trace::EnterCurrentFunction();
    size_t size;
    cl_mem mem;
    bool found = localHostAlloc_.translate(addr, mem, size);
    ASSERTION(found == true);

    cl_int ret;
    stream_t stream = cmd_.front();
    ret = clEnqueueUnmapMemObject(stream, mem, addr, 0, NULL, NULL);
    trace::ExitCurrentFunction();

    return error(ret);

}

}}}
