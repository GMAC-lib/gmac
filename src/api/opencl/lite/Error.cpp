#include "api/opencl/lite/Mode.h"

namespace __impl { namespace opencl { namespace lite {

#define __GMAC_ERROR(r, err) case r: error = err; break

gmacError_t
Mode::error(cl_int r) const
{
	gmacError_t error = gmacSuccess;
	switch(r) {
		__GMAC_ERROR(CL_SUCCESS, gmacSuccess);
        __GMAC_ERROR(CL_DEVICE_NOT_FOUND, gmacErrorNoAccelerator);
        __GMAC_ERROR(CL_DEVICE_NOT_AVAILABLE, gmacErrorInvalidAccelerator);
        __GMAC_ERROR(CL_MEM_OBJECT_ALLOCATION_FAILURE, gmacErrorMemoryAllocation);
        __GMAC_ERROR(CL_OUT_OF_HOST_MEMORY, gmacErrorMemoryAllocation);
        __GMAC_ERROR(CL_OUT_OF_RESOURCES, gmacErrorMemoryAllocation);
		default: error = gmacErrorUnknown;
	}
	return error;
}

#undef __GMAC_ERROR

}}}
