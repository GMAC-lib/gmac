#include <cuda.h>

#include <algorithm>
#include <list>
#include <string>

#include "api/cuda/hpe/Accelerator.h"
#include "api/cuda/hpe/Mode.h"

#include "config/order.h"
#include "core/Process.h"

#include "hpe/init.h"

#include "util/loader.h"
#include "util/Parameter.h"

static bool initialized = false;

void GMAC_API CUDA(gmac::core::hpe::Process &proc)
{
    TRACE(GLOBAL, "Initializing CUDA Driver API");
    if(initialized == false && cuInit(0) != CUDA_SUCCESS)
        FATAL("Unable to init CUDA");

    int devCount = 0;
    int devRealCount = 0;

    if(cuDeviceGetCount(&devCount) != CUDA_SUCCESS)
        FATAL("Error getting CUDA-enabled devices");

    TRACE(GLOBAL, "Found %d CUDA capable devices", devCount);

    std::list<int> visibleDevices;
    if (gmac::util::params::ParamVisibleDevices != NULL &&
        strlen(gmac::util::params::ParamVisibleDevices) > 0) {
        size_t len = strlen(gmac::util::params::ParamVisibleDevices);
        char *visible = new char[len + 1];
        ::memcpy(visible, gmac::util::params::ParamVisibleDevices, len + 1);

        const char *strDev = strtok(visible, ",");
        while (strDev != NULL) {
            int dev = atoi(strDev);
            visibleDevices.push_back(dev);
            strDev = strtok(NULL, ",");
        }

        delete []visible;
    }

    // Add accelerators to the system
    for(int i = 0; i < devCount; i++) {
        if (visibleDevices.size() > 0 &&
            std::find(visibleDevices.begin(), visibleDevices.end(), i) == visibleDevices.end()) continue;

        CUdevice cuDev;
        if(cuDeviceGet(&cuDev, i) != CUDA_SUCCESS)
            FATAL("Unable to access CUDA device");
#if CUDA_VERSION >= 2020
        int attr = 0;
        if(cuDeviceGetAttribute(&attr, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, cuDev) != CUDA_SUCCESS)
            FATAL("Unable to access CUDA device");

        if(attr == CU_COMPUTEMODE_PROHIBITED) {
	    TRACE(GLOBAL, "Device %d prohibited", i);
	} else {
            gmac::cuda::hpe::Accelerator *accelerator = new gmac::cuda::hpe::Accelerator(i, cuDev);
            CFATAL(accelerator != NULL, "Error allocating resources for the accelerator");
            proc.addAccelerator(*accelerator);
            devRealCount++;
        }
#else
        proc.addAccelerator(new gmac::cuda::Accelerator(i, cuDev));
        devRealCount++;
#endif
    }

    if(devRealCount == 0)
        FATAL("No CUDA-enabled devices found");

    // Initialize the private per-thread variables
    gmac::cuda::hpe::Accelerator::init();

    initialized = true;
}
