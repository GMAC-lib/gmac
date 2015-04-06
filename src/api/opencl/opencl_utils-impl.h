#ifndef GMAC_API_OPENCL_OPENCLUTILS_IMPL_H_
#define GMAC_API_OPENCL_OPENCLUTILS_IMPL_H_

#include <algorithm>

namespace __impl { namespace opencl { namespace util {

static std::string OpenCLNVidiaDevicePrefix[] = {
    "ION",
    "Tesla",
    "GeForce"
};

static std::string OpenCLAMDFusionDevice[] = {
    "Wrestler",
    "WinterPark",
    "BeaverCreek"
};

std::string
getPlatformString(int string, cl_platform_id id);

inline std::string
getPlatformName(cl_platform_id id)
{
    return getPlatformString(CL_PLATFORM_NAME, id);
}

inline std::string
getPlatformVendor(cl_platform_id id)
{
    return getPlatformString(CL_PLATFORM_VENDOR, id);
}

std::string
getDeviceString(int string, cl_device_id id);

inline std::string
getDeviceName(cl_device_id id)
{
    return getDeviceString(CL_DEVICE_NAME, id);
}

inline std::string
getDeviceVendor(cl_device_id id)
{
    return getDeviceString(CL_DEVICE_VENDOR, id);
}

inline bool
isDeviceAMDFusion(cl_device_id id)
{
    std::string *end = OpenCLAMDFusionDevice + 3;
    std::string *str = find(OpenCLAMDFusionDevice, end, getDeviceName(id));
    return str != end;
}

inline bool
isDeviceGPU(cl_device_id id)
{
    return getDeviceType(id) == CL_DEVICE_TYPE_GPU;
}

inline bool
isDeviceCPU(cl_device_id id)
{
    return getDeviceType(id) == CL_DEVICE_TYPE_CPU;
}

}}}

#endif
