#ifndef GMAC_API_OPENCL_OPENCLUTILS_H_
#define GMAC_API_OPENCL_OPENCLUTILS_H_

#include <string>

#include "config/common.h"

namespace __impl { namespace opencl { namespace util {

enum GMAC_LOCAL OpenCLVendor {
    VENDOR_AMD,
    VENDOR_NVIDIA,
    VENDOR_INTEL,
    VENDOR_UNKNOWN
};

enum GMAC_LOCAL OpenCLPlatform {
    PLATFORM_AMD,
    PLATFORM_NVIDIA,
    PLATFORM_INTEL,
    PLATFORM_APPLE,
    PLATFORM_UNKNOWN
};

typedef std::pair<unsigned, unsigned> OpenCLVersion;

// Platform functions
std::string GMAC_LOCAL
getPlatformName(cl_platform_id id);

std::string GMAC_LOCAL
getPlatformVendor(cl_platform_id id);

OpenCLPlatform GMAC_LOCAL
getPlatform(cl_platform_id id);

OpenCLVersion GMAC_LOCAL
getOpenCLVersion(cl_platform_id id);

// Device functions
cl_device_type GMAC_LOCAL
getDeviceType(cl_device_id id);

std::string GMAC_LOCAL
getDeviceName(cl_device_id id);

std::string GMAC_LOCAL
getDeviceVendor(cl_device_id id);

bool GMAC_LOCAL
isDeviceAMDFusion(cl_device_id id);

bool GMAC_LOCAL
isDeviceGPU(cl_device_id id);

bool GMAC_LOCAL
isDeviceCPU(cl_device_id id);

// Context functions
cl_device_id GMAC_LOCAL
getContextDevice(cl_context context);

std::vector<cl_device_id> GMAC_LOCAL
getContextDevices(cl_context context);

// Commmand queue functions
cl_device_id GMAC_LOCAL
getQueueDevice(cl_command_queue queue);

cl_context GMAC_LOCAL
getQueueContext(cl_command_queue queue);

}}}

#include "opencl_utils-impl.h"

#endif
