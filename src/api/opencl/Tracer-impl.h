#ifndef GMAC_API_OPENCL_TRACER_IMPL_H_
#define GMAC_API_OPENCL_TRACER_IMPL_H_

#include "api/opencl/Mode.h"
#include "core/Mode.h"

namespace __impl {

#if defined(USE_TRACE)
namespace trace {
extern Tracer *tracer;
}
#endif

namespace opencl {

#if defined(USE_TRACE)
inline
DataCommunication::DataCommunication() :
    stamp_(0), src_(0), dst_(0)
{}
#else
inline
DataCommunication::DataCommunication()
{}
#endif

inline
void DataCommunication::init(THREAD_T src, THREAD_T dst)
{
#if defined(USE_TRACE)
    if(trace::tracer == NULL) return;
    src_ = src; dst_ = dst;
    stamp_ = trace::tracer->timeMark();
#endif
}

inline
THREAD_T DataCommunication::getThreadId() const
{
#if defined(USE_TRACE)
    return trace::GetThreadId();
#else
    return THREAD_T(0);
#endif
}

inline
THREAD_T DataCommunication::getModeId(const Mode &mode) const
{
#if defined(USE_TRACE)
    return dynamic_cast<const core::Mode &>(mode).id();
#else
    return THREAD_T(0);
#endif
}

inline
void DataCommunication::trace(cl_event start, cl_event end, size_t size) const
{
#if defined(USE_TRACE)
    if(trace::tracer == NULL) return;

    uint64_t delay = 0, delta = 0;
    uint64_t queued = 0, started = 0, ended = 0;
    cl_int ret = clGetEventProfilingInfo(start, CL_PROFILING_COMMAND_QUEUED, sizeof(queued), &queued, NULL);
    if(ret != CL_SUCCESS) return;
    ret = clGetEventProfilingInfo(start, CL_PROFILING_COMMAND_START, sizeof(started), &started, NULL);
    if(ret != CL_SUCCESS) return;
    delay = (started - queued) / 1000;
    ret = clGetEventProfilingInfo(end, CL_PROFILING_COMMAND_END, sizeof(ended), &ended, NULL);
    if(ret != CL_SUCCESS) return;
    delta = (ended - started) / 1000;

    trace::tracer->dataCommunication(stamp_ + delay, src_, dst_, delta, size);
#endif
}


#if defined(USE_TRACE)
inline
KernelExecution::KernelExecution(unsigned major, unsigned minor) :
    stamp_(0), thread_(0), major_(major), minor_(minor)
{ }
#else
inline
KernelExecution::KernelExecution(unsigned /*major*/, unsigned /*minor*/)
{
}
#endif

inline
void KernelExecution::init(THREAD_T thread)
{
#if defined(USE_TRACE)
    if(trace::tracer == NULL || minor_ < 1) return;
    thread_ = thread;
    stamp_ = trace::tracer->timeMark();
#endif
}

inline
void KernelExecution::trace(cl_kernel kernel, cl_event event) const
{
#if defined(USE_TRACE)
    if(trace::tracer == NULL || minor_ < 1) return;
    TracePoint *data = new TracePoint();
    data->stamp = stamp_;
    data->thread = thread_;
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, TracePoint::NameSize, data->name, NULL);
    clSetEventCallback(event, CL_COMPLETE, KernelExecution::Callback, data);
#else
    clReleaseEvent(event);
#endif
}


#if defined(USE_TRACE)
inline
void KernelExecution::Callback(cl_event event, cl_int status, void *data)
{
    if(trace::tracer == NULL) return;
    TracePoint *point = (TracePoint *)data;
    int64_t delay = 0, delta = 0;
    cl_ulong queued = 0, started = 0, ended = 0;

    cl_int ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(queued), &queued, NULL);
    if(ret != CL_SUCCESS) return;
    ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(started), &started, NULL);
    if(ret != CL_SUCCESS) return;
    delay = (started - queued) / 1000;
        ASSERTION(delay > 0);
    ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(ended), &ended, NULL);
    if(ret != CL_SUCCESS) return;
    delta = (ended - started) / 1000;
        ASSERTION(delta > 0);
    clReleaseEvent(event);
        if(delta > 0) {
                trace::tracer->setThreadState(point->stamp + delay, point->thread, trace::Running);
                trace::tracer->setThreadState(point->stamp + delay + delta, point->thread, trace::Idle);

                trace::tracer->enterFunction(point->stamp + delay, point->thread, point->name);
                trace::tracer->exitFunction(point->stamp + delay + delta, point->thread, point->name);
        }
    delete point;
}
#endif

}}

#endif
