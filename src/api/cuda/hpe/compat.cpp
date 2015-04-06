#include <cuda.h>
#define __stdcall GMAC_API __stdcall
#include <driver_types.h>
#undef __stdcall

#include <string>
#include <list>

#include "api/cuda/hpe/Accelerator.h"
#include "api/cuda/hpe/Mode.h"
#include "api/cuda/hpe/Module.h"

#include "hpe/init.h"

#include "memory/Manager.h"
#include "core/hpe/Process.h"
#include "core/hpe/Thread.h"

using __impl::cuda::hpe::Switch;
using __impl::cuda::hpe::Texture;
using __impl::cuda::hpe::Variable;

static inline
__impl::cuda::hpe::Mode &getCurrentCUDAMode()
{
    return dynamic_cast<__impl::cuda::hpe::Mode &>(__impl::core::hpe::Thread::getCurrentMode());
}

static inline
int __getChannelSize(CUarray_format format)
{
	switch(format) {
		case CU_AD_FORMAT_SIGNED_INT8:
		case CU_AD_FORMAT_UNSIGNED_INT8:
			return 8;
		case CU_AD_FORMAT_SIGNED_INT16:
		case CU_AD_FORMAT_UNSIGNED_INT16:
		case CU_AD_FORMAT_HALF:
			return 16;
		case CU_AD_FORMAT_SIGNED_INT32:
		case CU_AD_FORMAT_UNSIGNED_INT32:
		case CU_AD_FORMAT_FLOAT:
			return 32;
	}
	return 0;
}

static inline CUarray_format __getChannelFormatKind(const struct cudaChannelFormatDesc *desc)
{
	int size = desc->x;
	switch(desc->f) {
		case cudaChannelFormatKindSigned:
			if(size == 8) return CU_AD_FORMAT_SIGNED_INT8;
			if(size == 16) return CU_AD_FORMAT_SIGNED_INT16;
			if(size == 32) return CU_AD_FORMAT_SIGNED_INT32;
		case cudaChannelFormatKindUnsigned:
			if(size == 8) return CU_AD_FORMAT_UNSIGNED_INT8;
			if(size == 16) return CU_AD_FORMAT_UNSIGNED_INT16;
			if(size == 32) return CU_AD_FORMAT_UNSIGNED_INT32;
		case cudaChannelFormatKindFloat:
			if(size == 16) return CU_AD_FORMAT_HALF;
			if(size == 32) return CU_AD_FORMAT_FLOAT;
		case cudaChannelFormatKindNone:
            break;
	};
	return CU_AD_FORMAT_UNSIGNED_INT32;
}

static inline unsigned int __getNumberOfChannels(const struct cudaChannelFormatDesc *desc)
{
	unsigned int n = 0;
	if(desc->x != 0) n++;
	if(desc->y != 0) n++;
	if(desc->z != 0) n++;
	if(desc->w != 0) n++;
	ASSERTION(n != 3);
	return n;
}

static inline void __setNumberOfChannels(struct cudaChannelFormatDesc *desc, unsigned int channels, int s)
{
	desc->x = desc->y = desc->z = desc->w = 0;
	if(channels >= 1) desc->x = s;
	if(channels >= 2) desc->y = s;
	if(channels >= 4) { desc->z = desc->w = s; }
}



static inline cudaChannelFormatKind __getCUDAChannelFormatKind(CUarray_format format)
{
	switch(format) {
		case CU_AD_FORMAT_UNSIGNED_INT8:
		case CU_AD_FORMAT_UNSIGNED_INT16:
		case CU_AD_FORMAT_UNSIGNED_INT32:
			return cudaChannelFormatKindUnsigned;
		case CU_AD_FORMAT_SIGNED_INT8:
		case CU_AD_FORMAT_SIGNED_INT16:
		case CU_AD_FORMAT_SIGNED_INT32:
			return cudaChannelFormatKindSigned;
		case CU_AD_FORMAT_HALF:
		case CU_AD_FORMAT_FLOAT:
			return cudaChannelFormatKindFloat;
	};
	return cudaChannelFormatKindSigned;
}


static inline cudaError_t __getCUDAError(CUresult r)
{
	switch(r) {
		case CUDA_SUCCESS:
			return cudaSuccess;
		case CUDA_ERROR_OUT_OF_MEMORY:
			return cudaErrorMemoryAllocation;
		case CUDA_ERROR_NOT_INITIALIZED:
		case CUDA_ERROR_DEINITIALIZED:
			return cudaErrorInitializationError;
        case CUDA_ERROR_INVALID_VALUE:
            return cudaErrorInvalidValue;
        case CUDA_ERROR_INVALID_DEVICE:
            return cudaErrorInvalidDevice;
        case CUDA_ERROR_MAP_FAILED:
            return cudaErrorMapBufferObjectFailed;
        case CUDA_ERROR_UNMAP_FAILED:
            return cudaErrorUnmapBufferObjectFailed;
        case CUDA_ERROR_LAUNCH_TIMEOUT:
            return cudaErrorLaunchTimeout;
        case CUDA_ERROR_LAUNCH_FAILED:
        case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
            return cudaErrorLaunchFailure;
        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
            return cudaErrorLaunchOutOfResources;
        case CUDA_ERROR_NO_DEVICE:
#if CUDA_VERSION >= 2020
            return cudaErrorNoDevice;
#endif
#if CUDA_VERSION >= 3000
        case CUDA_ERROR_ECC_UNCORRECTABLE:
            return cudaErrorECCUncorrectable;
#if CUDA_VERSION <= 3010
        case CUDA_ERROR_POINTER_IS_64BIT:
        case CUDA_ERROR_SIZE_IS_64BIT:
#endif
        case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
        case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
#endif
#if CUDA_VERSION >= 3010
        case CUDA_ERROR_UNSUPPORTED_LIMIT:
        case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
        case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
#endif
#if CUDA_VERSION >= 3020
        case CUDA_ERROR_OPERATING_SYSTEM:
#endif
#if CUDA_VERSION >= 4000
        case CUDA_ERROR_PROFILER_DISABLED:
        case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
        case CUDA_ERROR_PROFILER_ALREADY_STARTED:
        case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
        case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
        case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
        case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
        case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
        case CUDA_ERROR_CONTEXT_IS_DESTROYED:
#endif
#if CUDA_VERSION >= 4010
        case CUDA_ERROR_ASSERT:
        case CUDA_ERROR_TOO_MANY_PEERS:
        case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
        case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
#endif
        case CUDA_ERROR_ARRAY_IS_MAPPED:
        case CUDA_ERROR_ALREADY_MAPPED:
        case CUDA_ERROR_INVALID_CONTEXT:
        case CUDA_ERROR_INVALID_HANDLE:
        case CUDA_ERROR_INVALID_IMAGE:
        case CUDA_ERROR_INVALID_SOURCE:
        case CUDA_ERROR_FILE_NOT_FOUND:
        case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
        case CUDA_ERROR_NO_BINARY_FOR_GPU:
        case CUDA_ERROR_ALREADY_ACQUIRED:
        case CUDA_ERROR_NOT_FOUND:
        case CUDA_ERROR_NOT_READY:
        case CUDA_ERROR_NOT_MAPPED:
        case CUDA_ERROR_UNKNOWN:
            break;
	};
	return cudaErrorUnknown;
}


static inline CUmemorytype __getMemoryFrom(cudaMemcpyKind kind)
{
	switch(kind) {
		case cudaMemcpyHostToHost:
		case cudaMemcpyHostToDevice:
			return CU_MEMORYTYPE_HOST;
		case cudaMemcpyDeviceToHost:
		case cudaMemcpyDeviceToDevice:
			return CU_MEMORYTYPE_DEVICE;
        default:
            abort();
            return CU_MEMORYTYPE_DEVICE;
	}
}

static inline CUmemorytype __getMemoryTo(cudaMemcpyKind kind)
{
	switch(kind) {
		case cudaMemcpyHostToHost:
		case cudaMemcpyDeviceToHost:
			return CU_MEMORYTYPE_HOST;
		case cudaMemcpyHostToDevice:
		case cudaMemcpyDeviceToDevice:
			return CU_MEMORYTYPE_DEVICE;
        default:
            abort();
            return CU_MEMORYTYPE_DEVICE;
	}
}

#if CUDA_VERSION >= 3020
static cudaError_t __cudaMemcpyToArray(CUarray array, size_t wOffset,
		size_t hOffset, const void *src, size_t count)
#else
static cudaError_t __cudaMemcpyToArray(CUarray array, unsigned wOffset,
		unsigned hOffset, const void *src, unsigned count)
#endif
{
	CUDA_ARRAY_DESCRIPTOR desc;
    Switch::in(getCurrentCUDAMode());
	CUresult r = cuArrayGetDescriptor(&desc, array);
	if(r != CUDA_SUCCESS) return __getCUDAError(r);
#if CUDA_VERSION >= 3020
	size_t
#else
	unsigned
#endif
		offset = hOffset * desc.Width + wOffset;
	r = cuMemcpyHtoA(array, offset, src, count);
    Switch::out(getCurrentCUDAMode());
	return __getCUDAError(r);
}

#if CUDA_VERSION >= 3020
static cudaError_t __cudaMemcpy2D(CUarray dst, size_t wOffset, size_t hOffset,
		const void *src, size_t spitch, size_t width, size_t height)
#else
static cudaError_t __cudaMemcpy2D(CUarray dst, unsigned wOffset, unsigned hOffset,
		const void *src, unsigned spitch, unsigned width, unsigned height)
#endif
{
	TRACE(GLOBAL, "cudaMemcpy2DToArray ("FMT_SIZE" "FMT_SIZE" "FMT_SIZE ")", spitch, width, height);
	CUDA_MEMCPY2D cuCopy;
	memset(&cuCopy, 0, sizeof(cuCopy));

	cuCopy.srcMemoryType = CU_MEMORYTYPE_HOST;
	cuCopy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	cuCopy.srcHost = src;
	cuCopy.dstArray = (CUarray)dst;

	cuCopy.srcPitch = spitch;
	cuCopy.WidthInBytes = width;
	cuCopy.Height= height;

	cuCopy.srcXInBytes = wOffset;
	cuCopy.srcY = hOffset;

    Switch::in(getCurrentCUDAMode());
	CUresult r = cuMemcpy2D(&cuCopy);
    Switch::out(getCurrentCUDAMode());
	ASSERTION(r == CUDA_SUCCESS);
	return __getCUDAError(r);

}

#if CUDA_VERSION >= 3020
static cudaError_t __cudaInternalMemcpy2D(CUarray dst, size_t wOffset, size_t hOffset,
		CUdeviceptr src, size_t spitch, size_t width, size_t height)
#else
static cudaError_t __cudaInternalMemcpy2D(CUarray dst, unsigned wOffset, unsigned hOffset,
		CUdeviceptr src, unsigned spitch, unsigned width, unsigned height)
#endif
{
	TRACE(GLOBAL, "cudaMemcpy2DToArray ("FMT_SIZE" "FMT_SIZE " "FMT_SIZE")", spitch, width, height);
	CUDA_MEMCPY2D cuCopy;
	memset(&cuCopy, 0, sizeof(cuCopy));

	cuCopy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	cuCopy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	cuCopy.srcDevice = src;
	cuCopy.dstArray = dst;

	cuCopy.srcPitch = spitch;
	cuCopy.WidthInBytes = width;
	cuCopy.Height= height;

	cuCopy.srcXInBytes = wOffset;
	cuCopy.srcY = hOffset;

    Switch::in(getCurrentCUDAMode());
	CUresult r = cuMemcpy2DUnaligned(&cuCopy);
    Switch::out(getCurrentCUDAMode());
	ASSERTION(r == CUDA_SUCCESS);
	return __getCUDAError(r);

}

#ifdef __cplusplus
extern "C" {
#endif

GMAC_API cudaError_t APICALL cudaGetDeviceProperties(struct cudaDeviceProp *prop, int devNum)
{
    if (devNum < 0 || unsigned(devNum) >= __impl::core::hpe::getProcess().nAccelerators())
        return cudaErrorInvalidDevice;

    CUdevice device;
    CUresult ret;
    ret = cuDeviceGet(&device, devNum);
    if (ret != CUDA_SUCCESS) return __getCUDAError(ret);

    ret = cuDeviceGetName(prop->name, 255, device);
    if (ret != CUDA_SUCCESS) return __getCUDAError(ret);

    CUdevprop_st cuProps;
    ret = cuDeviceGetProperties(&cuProps, device);
    if (ret != CUDA_SUCCESS) return __getCUDAError(ret);

    ret = cuDeviceTotalMem(&prop->totalGlobalMem, device);
    if (ret != CUDA_SUCCESS) return __getCUDAError(ret);
 
    prop->sharedMemPerBlock = cuProps.sharedMemPerBlock;
    prop->regsPerBlock = cuProps.regsPerBlock;
    prop->warpSize = cuProps.SIMDWidth;
    prop->memPitch = cuProps.memPitch;
    prop->maxThreadsPerBlock = cuProps.maxThreadsPerBlock;
    prop->maxThreadsDim[0] = cuProps.maxThreadsDim[0];
    prop->maxThreadsDim[1] = cuProps.maxThreadsDim[1];
    prop->maxThreadsDim[2] = cuProps.maxThreadsDim[2];
    prop->maxGridSize[0] = cuProps.maxGridSize[0];
    prop->maxGridSize[1] = cuProps.maxGridSize[1];
    prop->maxGridSize[2] = cuProps.maxGridSize[2];
    prop->totalConstMem = cuProps.totalConstantMemory;

    ret = cuDeviceComputeCapability(&prop->major, &prop->minor, device);
    if (ret != CUDA_SUCCESS) return __getCUDAError(ret);

    prop->clockRate = cuProps.clockRate;
    prop->textureAlignment = cuProps.textureAlign;

    ret = cuDeviceGetAttribute(&prop->deviceOverlap, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, device);
    if (ret != CUDA_SUCCESS) return __getCUDAError(ret);

    ret = cuDeviceGetAttribute(&prop->multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
    if (ret != CUDA_SUCCESS) return __getCUDAError(ret);

    ret = cuDeviceGetAttribute(&prop->kernelExecTimeoutEnabled, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, device);
    if (ret != CUDA_SUCCESS) return __getCUDAError(ret);

    ret = cuDeviceGetAttribute(&prop->integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, device);
    if (ret != CUDA_SUCCESS) return __getCUDAError(ret);

    ret = cuDeviceGetAttribute(&prop->canMapHostMemory, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, device);
    if (ret != CUDA_SUCCESS) return __getCUDAError(ret);

    ret = cuDeviceGetAttribute(&prop->integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, device);
    if (ret != CUDA_SUCCESS) return __getCUDAError(ret);

    ret = cuDeviceGetAttribute(&prop->computeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, device);
    if (ret != CUDA_SUCCESS) return __getCUDAError(ret);

    ret = cuDeviceGetAttribute(&prop->concurrentKernels, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, device);
    if (ret != CUDA_SUCCESS) return __getCUDAError(ret);

    ret = cuDeviceGetAttribute(&prop->ECCEnabled, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, device);
    if (ret != CUDA_SUCCESS) return __getCUDAError(ret);

    ret = cuDeviceGetAttribute(&prop->pciBusID, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, device);
    if (ret != CUDA_SUCCESS) return __getCUDAError(ret);

    ret = cuDeviceGetAttribute(&prop->pciDeviceID, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device);
    if (ret != CUDA_SUCCESS) return __getCUDAError(ret);

    ret = cuDeviceGetAttribute(&prop->tccDriver, CU_DEVICE_ATTRIBUTE_TCC_DRIVER, device);
    if (ret != CUDA_SUCCESS) return __getCUDAError(ret);

    return cudaSuccess;
}

// CUDA Channel related functions

GMAC_API struct cudaChannelFormatDesc APICALL cudaCreateChannelDesc(int x, int y, int z,
		int w, enum cudaChannelFormatKind kind)
{
	struct cudaChannelFormatDesc desc;
	desc.x = x; desc.y = y; desc.z = z; desc.w = w;
	desc.f = kind;
	return desc;
}

GMAC_API cudaError_t APICALL cudaGetChannelDesc(struct cudaChannelFormatDesc *desc,
		const struct cudaArray *array)
{
	enterGmac();
	CUDA_ARRAY_DESCRIPTOR cuDesc;
    Switch::in(getCurrentCUDAMode());
	CUresult r = cuArrayGetDescriptor(&cuDesc, (CUarray)array);
    Switch::out(getCurrentCUDAMode());
	if(r != CUDA_SUCCESS) {
		exitGmac();
		return __getCUDAError(r);
	}
	desc->f = __getCUDAChannelFormatKind(cuDesc.Format);
	__setNumberOfChannels(desc, cuDesc.NumChannels, __getChannelSize(cuDesc.Format));
	TRACE(GLOBAL, "cudaGetChannelDesc %d %d %d %d %d", desc->x, desc->y, desc->z,
		desc->w, desc->f);
	exitGmac();
	return cudaSuccess;
}

// CUDA Array related functions

#if CUDA_VERSION >= 3010
GMAC_API cudaError_t APICALL cudaMallocArray(struct cudaArray **array,
		const struct cudaChannelFormatDesc *desc, size_t width,
		size_t height, unsigned int /*flags*/)
#else
GMAC_API cudaError_t APICALL cudaMallocArray(struct cudaArray **array,
		const struct cudaChannelFormatDesc *desc, size_t width,
		size_t height)
#endif
{
	CUDA_ARRAY_DESCRIPTOR cuDesc;
#if CUDA_VERSION >= 3020
	cuDesc.Width = width;
	cuDesc.Height = height;
#else
	cuDesc.Width = unsigned(width);
	cuDesc.Height = unsigned(height);
#endif
	cuDesc.Format = __getChannelFormatKind(desc);
	cuDesc.NumChannels = __getNumberOfChannels(desc);
	TRACE(GLOBAL, "cudaMallocArray: "FMT_SIZE" "FMT_SIZE" with format 0x%x and %u channels",
			width, height, cuDesc.Format, cuDesc.NumChannels);
	enterGmac();
    Switch::in(getCurrentCUDAMode());
	CUresult r = cuArrayCreate((CUarray *)array, &cuDesc);
    Switch::out(getCurrentCUDAMode());
	exitGmac();
	return __getCUDAError(r);
}

GMAC_API cudaError_t APICALL cudaFreeArray(struct cudaArray *array)
{
	enterGmac();
    Switch::in(getCurrentCUDAMode());
	CUresult r = cuArrayDestroy((CUarray)array);
    Switch::out(getCurrentCUDAMode());
	exitGmac();
	return __getCUDAError(r);
}

GMAC_API cudaError_t APICALL cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset,
		size_t hOffset, const void *src, size_t count,
		enum cudaMemcpyKind kind)
{
	ASSERTION(kind == cudaMemcpyHostToDevice);
	enterGmac();
#if CUDA_VERSION >= 3020
	cudaError_t ret = __cudaMemcpyToArray((CUarray)dst, wOffset, hOffset, src, count);
#else
	cudaError_t ret = __cudaMemcpyToArray((CUarray)dst, unsigned(wOffset),
                                                        unsigned(hOffset), src, unsigned(count));
#endif
	exitGmac();
	return ret;
}

GMAC_API cudaError_t APICALL cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset,
		size_t hOffset, const void *src, size_t spitch, size_t width,
		size_t height, enum cudaMemcpyKind kind)
{
	ASSERTION(kind == cudaMemcpyHostToDevice);
	enterGmac();
	cudaError_t ret = cudaSuccess;
    __impl::core::Process &proc = __impl::core::getProcess();
    __impl::cuda::Mode *mode = dynamic_cast<__impl::cuda::Mode *>(proc.owner(hostptr_t(src)));
    if(mode == NULL) {
#if CUDA_VERSION >= 3020
        __cudaMemcpy2D((CUarray)dst, wOffset, hOffset, src, spitch, width, height);
#else
        __cudaMemcpy2D((CUarray)dst, unsigned(wOffset), unsigned(hOffset), src,
                                     unsigned(spitch),  unsigned(width),
                                     unsigned(height));
#endif
    }
    else {
        long_t dummy = proc.translate(hostptr_t(src));
#if CUDA_VERSION >= 3020
        __cudaInternalMemcpy2D((CUarray)dst, wOffset, hOffset, (CUdeviceptr)(dummy), spitch, width, height);
#else
        __cudaInternalMemcpy2D((CUarray)dst, unsigned(wOffset), unsigned(hOffset), (CUdeviceptr)(dummy),
                                             unsigned(spitch),  unsigned(width), unsigned(height));
#endif
    }
	exitGmac();
	return ret;
}

#ifdef __cplusplus
}
#endif


// Functions related to constant memory

#ifdef __cplusplus
extern "C" {
#endif

GMAC_API cudaError_t APICALL cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count,
		size_t offset, enum cudaMemcpyKind kind)
{
	enterGmac();
	cudaError_t ret = cudaSuccess;
    __impl::cuda::hpe::Mode &mode = getCurrentCUDAMode();
	const Variable *variable = mode.constant(symbol);
	ASSERTION(variable != NULL);
	CUresult r = CUDA_SUCCESS;
	ASSERTION(variable->size() >= (count + offset));
	CUdeviceptr ptr = CUdeviceptr(variable->devPtr() + offset);
	switch(kind) {
		case cudaMemcpyHostToDevice:
			TRACE(GLOBAL, "cudaMemcpyToSymbol HostToDevice %p to 0x%x ("FMT_SIZE" bytes)", src, ptr, count);
            Switch::in(getCurrentCUDAMode());
#if CUDA_VERSION >= 3020
            r = cuMemcpyHtoD(ptr, src, count);
#else
            r = cuMemcpyHtoD(ptr, src, unsigned(count));
#endif
            Switch::out(getCurrentCUDAMode());
            ret = __getCUDAError(r);
            break;

		default:
			abort();
	}
    exitGmac();
    return ret;
}

#ifdef __cplusplus
}
#endif


// CUDA Texture related functions

static inline CUfilter_mode __getFilterMode(cudaTextureFilterMode mode)
{
	switch(mode) {
		case cudaFilterModePoint:
			return CU_TR_FILTER_MODE_POINT;
		case cudaFilterModeLinear:
			return CU_TR_FILTER_MODE_LINEAR;
		default:
			return CU_TR_FILTER_MODE_LINEAR;
	};
}

static inline CUaddress_mode __getAddressMode(cudaTextureAddressMode mode)
{
	switch(mode) {
		case cudaAddressModeWrap:
			return CU_TR_ADDRESS_MODE_WRAP;
		case cudaAddressModeClamp:
			return CU_TR_ADDRESS_MODE_CLAMP;
        default:
			return CU_TR_ADDRESS_MODE_WRAP;
	};
}

#ifdef __cplusplus
extern "C" {
#endif

GMAC_API cudaError_t APICALL cudaBindTexture(size_t *offset, const struct textureReference *texref,
     const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size = UINT_MAX)
{
    enterGmac();
    __impl::cuda::hpe::Mode &mode = getCurrentCUDAMode();
	CUresult r;
    const Texture * texture = mode.texture(texref);
    Switch::in(getCurrentCUDAMode());
	for(int i = 0; i < 3; i++) {
		r = cuTexRefSetAddressMode(texture->texRef(), i, __getAddressMode(texref->addressMode[i]));
		if(r != CUDA_SUCCESS) {
            Switch::out(getCurrentCUDAMode());
			exitGmac();
			return __getCUDAError(r);
		}
	}
	r = cuTexRefSetFlags(texture->texRef(), CU_TRSF_READ_AS_INTEGER);
	if(r != CUDA_SUCCESS) {
        Switch::out(getCurrentCUDAMode());
		exitGmac();
		return __getCUDAError(r);
	}
	r = cuTexRefSetFilterMode(texture->texRef(), __getFilterMode(texref->filterMode));
	if(r != CUDA_SUCCESS) {
        Switch::out(getCurrentCUDAMode());
		exitGmac();
		return __getCUDAError(r);
	}
	r = cuTexRefSetFormat(texture->texRef(),
			__getChannelFormatKind(&texref->channelDesc),
			__getNumberOfChannels(&texref->channelDesc));
	if(r != CUDA_SUCCESS) {
        Switch::out(getCurrentCUDAMode());
		exitGmac();
		return __getCUDAError(r);
	}
	r = cuTexRefSetAddress(offset, texture->texRef(), CUdeviceptr(devPtr), size);

    Switch::out(getCurrentCUDAMode());
	exitGmac();
	return __getCUDAError(r);
}


GMAC_API cudaError_t APICALL cudaBindTextureToArray(const struct textureReference *texref,
		const struct cudaArray *array, const struct cudaChannelFormatDesc * /*desc*/)
{
	enterGmac();
    __impl::cuda::hpe::Mode &mode = getCurrentCUDAMode();
	CUresult r;
    const Texture * texture = mode.texture(texref);
    Switch::in(getCurrentCUDAMode());
	for(int i = 0; i < 3; i++) {
		r = cuTexRefSetAddressMode(texture->texRef(), i, __getAddressMode(texref->addressMode[i]));
		if(r != CUDA_SUCCESS) {
            Switch::out(getCurrentCUDAMode());
			exitGmac();
			return __getCUDAError(r);
		}
	}
	r = cuTexRefSetFlags(texture->texRef(), CU_TRSF_READ_AS_INTEGER);
	if(r != CUDA_SUCCESS) {
        Switch::out(getCurrentCUDAMode());
		exitGmac();
		return __getCUDAError(r);
	}
	r = cuTexRefSetFilterMode(texture->texRef(), __getFilterMode(texref->filterMode));
	if(r != CUDA_SUCCESS) {
        Switch::out(getCurrentCUDAMode());
		exitGmac();
		return __getCUDAError(r);
	}
	r = cuTexRefSetFormat(texture->texRef(),
			__getChannelFormatKind(&texref->channelDesc),
			__getNumberOfChannels(&texref->channelDesc));
	if(r != CUDA_SUCCESS) {
        Switch::out(getCurrentCUDAMode());
		exitGmac();
		return __getCUDAError(r);
	}
	r = cuTexRefSetArray(texture->texRef(), (CUarray)array, CU_TRSA_OVERRIDE_FORMAT);

    Switch::out(getCurrentCUDAMode());
	exitGmac();
	return __getCUDAError(r);
}

GMAC_API cudaError_t APICALL cudaUnbindTexture(const struct textureReference *texref)
{
	enterGmac();
    __impl::cuda::hpe::Mode &mode = getCurrentCUDAMode();
    const Texture * texture = mode.texture(texref);
    Switch::in(getCurrentCUDAMode());
	CUresult r = cuTexRefDestroy(texture->texRef());
    Switch::out(getCurrentCUDAMode());
	exitGmac();
	return __getCUDAError(r);
}

GMAC_API void APICALL __cudaTextureFetch(const void * /*tex*/, void * /*index*/, int /*integer*/, void * /*val*/)
{
	ASSERTION(0);
}

GMAC_API int APICALL __cudaSynchronizeThreads(void**, void*)
{
	ASSERTION(0);
    return 0;
}

GMAC_API void APICALL __cudaMutexOperation(int /*lock*/)
{
	ASSERTION(0);
}


// Events and other stuff needed by CUDA Wrapper
GMAC_API cudaError_t APICALL cudaEventCreate(cudaEvent_t *event)
{
#if CUDA_VERSION >= 2020
    CUresult ret = cuEventCreate((CUevent *)event, CU_EVENT_DEFAULT);
#else
    CUresult ret = cuEventCreate((CUevent *)event, 0);
#endif
    return __getCUDAError(ret);
}

GMAC_API cudaError_t APICALL cudaEventDestroy(cudaEvent_t event)
{
    CUresult ret = cuEventDestroy((CUevent) event);
    return __getCUDAError(ret);
}

GMAC_API cudaError_t APICALL cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    CUresult ret = cuEventElapsedTime(ms, (CUevent)start, (CUevent)end);
    return __getCUDAError(ret);
}

GMAC_API cudaError_t APICALL cudaEventQuery(cudaEvent_t event)
{
    CUresult ret = cuEventQuery((CUevent) event);
    return __getCUDAError(ret);
}

GMAC_API cudaError_t APICALL cudaEventRecord(cudaEvent_t event, cudaStream_t /*stream*/)
{
    CUresult ret = cuEventRecord((CUevent) event, getCurrentCUDAMode().eventStream());
    return __getCUDAError(ret);
}

GMAC_API cudaError_t APICALL cudaEventSynchronize(cudaEvent_t event)
{
    CUresult ret = cuEventSynchronize((CUevent) event);
    return __getCUDAError(ret);
}



#ifdef __cplusplus
}
#endif

