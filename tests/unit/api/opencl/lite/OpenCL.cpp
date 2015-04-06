#include "OpenCL.h"


bool CreateOpenCLContext(cl_device_id &device, cl_context &context)
{
    cl_int error_code;
    cl_platform_id platform;
    
    error_code = clGetPlatformIDs(1, &platform, NULL);
    if(error_code != CL_SUCCESS) return false;

    error_code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);
    if(error_code != CL_SUCCESS) return false;

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &error_code);
    if(error_code != CL_SUCCESS) return false;
    
    return true;
}
