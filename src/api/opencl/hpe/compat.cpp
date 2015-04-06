#if defined(__APPLE__)
#   include <OpenCL/cl.h>
#   include <OpenCL/cl_gl.h>
#else
#   include <CL/cl.h>
#   include <CL/cl_gl.h>
#endif

#include "api/opencl/hpe/Mode.h"

#include "core/hpe/Thread.h"

static inline __impl::opencl::hpe::Mode &getCurrentCLMode()
{
    return dynamic_cast<__impl::opencl::hpe::Mode &>(__impl::core::hpe::Thread::getCurrentMode());
}


#ifdef __cplusplus
extern "C" {
#endif

GMAC_API cl_mem APICALL
oclCreateFromGLBuffer(cl_mem_flags flags, cl_GLuint bufobj, cl_int *errcode_ret)
{
	enterGmac();
    __impl::opencl::hpe::Mode &mode = getCurrentCLMode();
    cl_context context = mode.getAccelerator().getCLContext();
    cl_mem ret = clCreateFromGLBuffer(context, flags, bufobj, errcode_ret);
	exitGmac();

    return ret;
}

#ifdef USE_DEPRECATED_OPENCL_1_1
GMAC_API cl_mem APICALL
oclCreateFromGLTexture2D(cl_mem_flags flags, cl_GLenum texture_target, cl_GLint miplevel, cl_GLuint texture, cl_int *errcode_ret)
{
    enterGmac();
    __impl::opencl::hpe::Mode &mode = getCurrentCLMode();
    cl_context context = mode.getAccelerator().getCLContext();
    cl_mem ret = clCreateFromGLTexture2D(context, flags, texture_target, miplevel, texture, errcode_ret);
	exitGmac();

    return ret;
}
#endif

#ifdef USE_DEPRECATED_OPENCL_1_1
GMAC_API cl_mem APICALL
oclCreateFromGLTexture3D(cl_mem_flags flags, cl_GLenum texture_target, cl_GLint miplevel, cl_GLuint texture, cl_int *errcode_ret)
{
    enterGmac();
    __impl::opencl::hpe::Mode &mode = getCurrentCLMode();
    cl_context context = mode.getAccelerator().getCLContext();
    cl_mem ret = clCreateFromGLTexture3D(context, flags, texture_target, miplevel, texture, errcode_ret);
	exitGmac();

    return ret;
}
#endif

GMAC_API cl_mem APICALL
oclCreateFromGLRenderbuffer(cl_mem_flags flags, cl_GLuint renderbuffer, cl_int *errcode_ret) 
{
    enterGmac();
    __impl::opencl::hpe::Mode &mode = getCurrentCLMode();
    cl_context context = mode.getAccelerator().getCLContext();
    cl_mem ret = clCreateFromGLRenderbuffer(context, flags, renderbuffer, errcode_ret);
	exitGmac();

    return ret;
}

GMAC_API cl_int APICALL
oclEnqueueAcquireGLObjects(cl_uint num_objects, const cl_mem *mem_objects,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
{
    enterGmac();
    __impl::opencl::hpe::Mode &mode = getCurrentCLMode();
    cl_command_queue queue = mode.eventStream();
    cl_int ret = clEnqueueAcquireGLObjects(queue, num_objects, mem_objects, num_events_in_wait_list, event_wait_list, event);
    if (ret == CL_SUCCESS) clFlush(queue);
	exitGmac();

    return ret;
}

GMAC_API cl_int APICALL
oclAcquireGLObjects(cl_uint num_objects, const cl_mem *mem_objects)
{
    enterGmac();
    __impl::opencl::hpe::Mode &mode = getCurrentCLMode();
    cl_command_queue queue = mode.eventStream();
    cl_int ret = clEnqueueAcquireGLObjects(queue, num_objects, mem_objects, 0, NULL, NULL);
    if (ret == CL_SUCCESS) clFinish(queue);
	exitGmac();

    return ret;
}

GMAC_API cl_int APICALL
oclEnqueueReleaseGLObjects(cl_uint num_objects,  const cl_mem *mem_objects,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
{
    enterGmac();
    __impl::opencl::hpe::Mode &mode = getCurrentCLMode();
    cl_command_queue queue = mode.eventStream();
    cl_int ret = clEnqueueReleaseGLObjects(queue, num_objects, mem_objects, num_events_in_wait_list, event_wait_list, event);
    if (ret == CL_SUCCESS) clFlush(queue);
	exitGmac();

    return ret;
}

GMAC_API cl_int APICALL
oclReleaseGLObjects(cl_uint num_objects,  const cl_mem *mem_objects)
{
    enterGmac();
    __impl::opencl::hpe::Mode &mode = getCurrentCLMode();
    cl_command_queue queue = mode.eventStream();
    cl_int ret = clEnqueueReleaseGLObjects(queue, num_objects, mem_objects, 0, NULL, NULL);
    if (ret == CL_SUCCESS) clFinish(queue);
	exitGmac();

    return ret;
}

#ifdef USE_KHR_EXTENSIONS
GMAC_API cl_event APICALL
oclCreateEventFromGLsyncKHR(cl_GLsync sync, cl_int *errcode_ret)
{
    enterGmac();
    __impl::opencl::hpe::Mode &mode = getCurrentCLMode();
    cl_context context = mode.getAccelerator().getCLContext();
    cl_event ret = clCreateEventFromGLsyncKHR(context, sync, errcode_ret);
	exitGmac();

    return ret;
}
#endif


#ifdef __cplusplus
}
#endif


/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
