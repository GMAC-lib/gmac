#include "config/order.h"

#include "core/hpe/Process.h"
#include "cpu/Accelerator.h"
#include "gpu/amd/Accelerator.h"
#include "gpu/amd/FusionAccelerator.h"
#include "gpu/nvidia/Accelerator.h"

#include "api/opencl/opencl_utils.h"

static bool initialized = false;
void OpenCL(gmac::core::hpe::Process &proc)
{
    TRACE(GLOBAL, "Initializing OpenCL API");
    cl_uint platformSize = 0;
    cl_int ret = CL_SUCCESS;
    ret = clGetPlatformIDs(0, NULL, &platformSize);
    CFATAL(ret == CL_SUCCESS);
    if(platformSize == 0) return;   
    cl_platform_id * platforms = new cl_platform_id[platformSize];
    ret = clGetPlatformIDs(platformSize, platforms, NULL);
    CFATAL(ret == CL_SUCCESS);
    MESSAGE("%d OpenCL platforms found", platformSize);

    unsigned n = 0;
    for (unsigned i = 0; i < platformSize; i++) {
        MESSAGE("Platform [%u/%u]: %s", i + 1, platformSize, __impl::opencl::util::getPlatformName(platforms[i]).c_str());
        cl_uint deviceSize = 0;
        cl_device_type types = CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU;
        //cl_device_type types = CL_DEVICE_TYPE_CPU;
        ret = clGetDeviceIDs(platforms[i], types,
                             0, NULL, &deviceSize);
        ASSERTION(ret == CL_SUCCESS);
	    if(deviceSize == 0) continue;
        cl_device_id *devices = new cl_device_id[deviceSize];
        ret = clGetDeviceIDs(platforms[i], types,
                             deviceSize, devices, NULL);
        ASSERTION(ret == CL_SUCCESS);
        MESSAGE("... found %u OpenCL devices", deviceSize, i);

        __impl::opencl::util::OpenCLVersion clVersion = __impl::opencl::util::getOpenCLVersion(platforms[i]);

        cl_context ctx;
        if (deviceSize > 0) {
            cl_context_properties prop[] = {
                CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i], 0 };

            ctx = clCreateContext(prop, deviceSize, devices, NULL, NULL, &ret);
            CFATAL(ret == CL_SUCCESS, "Unable to create OpenCL context %d", ret);
        }

        for (unsigned j = 0; j < deviceSize; j++) {
            MESSAGE("Device [%u/%u]: %s", j + 1, deviceSize, __impl::opencl::util::getDeviceName(devices[j]).c_str());

            // Let's assume that this is not important; TODO: actually deal with this case
            //CFATAL(__impl::opencl::util::getDeviceVendor(devices[j]) == __impl::opencl::util::getPlatformVendor(platforms[i]), "Not handled case");
            gmac::opencl::hpe::Accelerator *acc = NULL;

            switch (__impl::opencl::util::getPlatform(platforms[i])) {
                case __impl::opencl::util::PLATFORM_AMD:
                    if (__impl::opencl::util::isDeviceAMDFusion(devices[j])) {
#if 0
                        // Enable when the API stabilizes a little bit and we can use user-provided fonts
                        acc = new __impl::opencl::hpe::gpu::amd::FusionAccelerator(n++, ctx, devices[j],
                                clVersion.first, clVersion.second);
#else
                        acc = new __impl::opencl::hpe::gpu::amd::Accelerator(n++, ctx, devices[j],
                                clVersion.first, clVersion.second);
#endif
                    } else if (__impl::opencl::util::isDeviceGPU(devices[j])) {
                        acc = new __impl::opencl::hpe::gpu::amd::Accelerator(n++, ctx, devices[j],
                                clVersion.first, clVersion.second);
                    } else if (__impl::opencl::util::isDeviceCPU(devices[j])) {
                        acc = new __impl::opencl::hpe::cpu::Accelerator(n++, ctx, devices[j],
                                clVersion.first, clVersion.second);
                    } else {
                        FATAL("Not supported device");
                    }
                    break;
                case __impl::opencl::util::PLATFORM_APPLE:
                case __impl::opencl::util::PLATFORM_NVIDIA:
                    acc = new __impl::opencl::hpe::gpu::nvidia::Accelerator(n++, ctx, devices[j],
                            clVersion.first, clVersion.second);
                    break;
                case __impl::opencl::util::PLATFORM_INTEL:
                case __impl::opencl::util::PLATFORM_UNKNOWN:
                    FATAL("Platform not supported");
            }
            proc.addAccelerator(*acc);
            // Nedded for OpenCL code compilation
            __impl::opencl::hpe::Accelerator::addAccelerator(*acc);
        }
        if (deviceSize > 0) {
            ret = clReleaseContext(ctx);
            CFATAL(ret == CL_SUCCESS, "Unable to release OpenCL context after accelerator initialization");
        }
        delete[] devices;
    }
    delete[] platforms;
    initialized = true;

    __impl::opencl::hpe::Accelerator::prepareEmbeddedCLCode();
}
