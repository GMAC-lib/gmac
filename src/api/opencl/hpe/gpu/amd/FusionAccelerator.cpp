#include "api/opencl/IOBuffer.h"

#include "FusionAccelerator.h"

namespace __impl { namespace opencl { namespace hpe { namespace gpu { namespace amd {

FusionAccelerator::FusionAccelerator(int n, cl_context context, cl_device_id device, unsigned major, unsigned minor) :
    gmac::opencl::hpe::Accelerator(n, context, device, major, minor)
{
    integrated_ = true;
}

FusionAccelerator::~FusionAccelerator()
{
}

gmacError_t FusionAccelerator::copyToAcceleratorAsync(accptr_t acc, core::IOBuffer &_buffer,
    size_t bufferOff, size_t count, core::hpe::Mode &mode, cl_command_queue stream)
{
    IOBuffer &buffer = dynamic_cast<IOBuffer &>(_buffer);
    hostptr_t host = buffer.addr() + bufferOff;

    trace::EnterCurrentFunction();
    TRACE(LOCAL, "Async copy to accelerator: %p ("FMT_SIZE") @ %p", host, count, acc.get());

    cl_event start, end;
    cl_int ret;

    if (buffer.async() == true) {
        cl_mem mem = buffer.getCLBuffer();

        buffer.toAccelerator(dynamic_cast<opencl::Mode &>(mode));

        lock();
        cl_int err = clEnqueueUnmapMemObject(stream, mem, buffer.addr(),
                0, NULL, &start);
        ASSERTION(err == CL_SUCCESS);

        ret = clEnqueueCopyBuffer(stream, mem, acc.get(), bufferOff,
                acc.offset(), count, 0, NULL, NULL);
        CFATAL(ret == CL_SUCCESS, "Error copying to accelerator: %d", ret);
        cl_int flags;
        if (buffer.getProtection() == GMAC_PROT_WRITE) {
            flags = CL_MAP_WRITE;
        } else {
            flags = CL_MAP_READ | CL_MAP_WRITE;
        }
        hostptr_t addr = (hostptr_t)clEnqueueMapBuffer(stream, mem, CL_FALSE,
                                                       flags, 0, buffer.size(), 0, NULL, &end, &err);
        ASSERTION(err == CL_SUCCESS);
        unlock();

        buffer.started(start, end, count);
        buffer.setAddr(addr);
        ret = clFlush(stream);
        CFATAL(ret == CL_SUCCESS, "Error issuing copy to accelerator: %d", ret);
    } else {
        uint8_t *host = buffer.addr() + bufferOff;

        buffer.toAccelerator(dynamic_cast<opencl::Mode &>(mode));
        lock();
        ret = clEnqueueWriteBuffer(stream, acc.get(), CL_FALSE,
                acc.offset(), count, host, 0, NULL, &start);
        CFATAL(ret == CL_SUCCESS, "Error copying to accelerator: %d", ret);
        unlock();

        buffer.started(start, count);
        ret = clFlush(stream);
        CFATAL(ret == CL_SUCCESS, "Error issuing copy to accelerator: %d", ret);
    }

    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t FusionAccelerator::copyToHostAsync(core::IOBuffer &_buffer, size_t bufferOff,
    const accptr_t acc, size_t count, core::hpe::Mode &mode, cl_command_queue stream)
{
    IOBuffer &buffer = dynamic_cast<IOBuffer &>(_buffer);
    hostptr_t host = buffer.addr() + bufferOff;

    trace::EnterCurrentFunction();
    TRACE(LOCAL, "Async copy to host: %p ("FMT_SIZE") @ %p", host, count, acc.get());
    cl_event start, end;
    cl_int ret;

    if (buffer.async() == true) {
        cl_mem mem = buffer.getCLBuffer();

        buffer.toHost(reinterpret_cast<opencl::hpe::Mode &>(mode));
        lock();
        cl_int err = clEnqueueUnmapMemObject(stream, mem, buffer.addr(),
                0, NULL, &start);

        ASSERTION(err == CL_SUCCESS);

        ret = clEnqueueCopyBuffer(stream, acc.get(), mem,
                acc.offset(), bufferOff, count, 0, NULL, NULL);
        CFATAL(ret == CL_SUCCESS, "Error copying to host: %d", ret);
        cl_int flags;
        if (buffer.getProtection() == GMAC_PROT_WRITE) {
            flags = CL_MAP_WRITE;
        } else {
            flags = CL_MAP_READ | CL_MAP_WRITE;
        }
        hostptr_t addr = (hostptr_t)clEnqueueMapBuffer(stream, mem, CL_FALSE,
                                                       flags, 0, buffer.size(), 0, NULL, &end, &err);
        unlock();
        ASSERTION(err == CL_SUCCESS);

        buffer.started(start, end, count);
        buffer.setAddr(addr);
        ret = clFlush(stream);
        CFATAL(ret == CL_SUCCESS, "Error issuing read to accelerator: %d", ret);
    } else {
        uint8_t *host = buffer.addr() + bufferOff;

        buffer.toHost(reinterpret_cast<opencl::hpe::Mode &>(mode));
        lock();
        ret = clEnqueueReadBuffer(stream, acc.get(), CL_FALSE,
                acc.offset(), count, host, 0, NULL, &start);
        CFATAL(ret == CL_SUCCESS, "Error copying to host: %d", ret);
        unlock();
        buffer.started(start, count);
        ret = clFlush(stream);
        CFATAL(ret == CL_SUCCESS, "Error issuing read to accelerator: %d", ret);
    }

    trace::ExitCurrentFunction();
    return error(ret);
}

}}}}}
