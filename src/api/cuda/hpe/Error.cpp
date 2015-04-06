#include "Accelerator.h"

namespace __impl { namespace cuda { namespace hpe {

#define __GMAC_ERROR(r, err) case r: error = err; break

gmacError_t
Accelerator::error(CUresult r)
{
	gmacError_t error = gmacSuccess;
	switch(r) {
		__GMAC_ERROR(CUDA_SUCCESS, gmacSuccess);
		__GMAC_ERROR(CUDA_ERROR_INVALID_VALUE, gmacErrorInvalidValue);
		__GMAC_ERROR(CUDA_ERROR_OUT_OF_MEMORY, gmacErrorMemoryAllocation);
		__GMAC_ERROR(CUDA_ERROR_NOT_INITIALIZED, gmacErrorNotReady);
		__GMAC_ERROR(CUDA_ERROR_DEINITIALIZED, gmacErrorNotReady);
		__GMAC_ERROR(CUDA_ERROR_NO_DEVICE, gmacErrorNoAccelerator);
		__GMAC_ERROR(CUDA_ERROR_INVALID_DEVICE, gmacErrorInvalidAccelerator);
		__GMAC_ERROR(CUDA_ERROR_INVALID_IMAGE, gmacErrorInvalidAcceleratorFunction);
		__GMAC_ERROR(CUDA_ERROR_INVALID_CONTEXT, gmacErrorApiFailureBase);
		__GMAC_ERROR(CUDA_ERROR_CONTEXT_ALREADY_CURRENT, gmacErrorApiFailureBase);
		__GMAC_ERROR(CUDA_ERROR_ALREADY_MAPPED, gmacErrorMemoryAllocation);
		__GMAC_ERROR(CUDA_ERROR_NO_BINARY_FOR_GPU, gmacErrorInvalidAcceleratorFunction);	
		__GMAC_ERROR(CUDA_ERROR_ALREADY_ACQUIRED, gmacErrorApiFailureBase);
		__GMAC_ERROR(CUDA_ERROR_FILE_NOT_FOUND, gmacErrorApiFailureBase);
		__GMAC_ERROR(CUDA_ERROR_INVALID_HANDLE, gmacErrorApiFailureBase);
		__GMAC_ERROR(CUDA_ERROR_NOT_FOUND, gmacErrorApiFailureBase);
		__GMAC_ERROR(CUDA_ERROR_NOT_READY, gmacErrorNotReady);
		__GMAC_ERROR(CUDA_ERROR_LAUNCH_FAILED, gmacErrorLaunchFailure);	
		__GMAC_ERROR(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, gmacErrorLaunchFailure);
		__GMAC_ERROR(CUDA_ERROR_LAUNCH_TIMEOUT, gmacErrorLaunchFailure);
		__GMAC_ERROR(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING, gmacErrorLaunchFailure);
		__GMAC_ERROR(CUDA_ERROR_UNKNOWN, gmacErrorUnknown);
		default: error = gmacErrorUnknown;
	}
	return error;
}

#undef __GMAC_ERROR

}}}
