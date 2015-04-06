#include <cstring>

#include <vector_types.h>
#include <driver_types.h>

#include <string>
#include <vector>

#include "include/gmac/cuda.h"
#include "hpe/init.h"

#include "core/hpe/Kernel.h"
#include "core/hpe/Thread.h"

#include "api/cuda/hpe/Accelerator.h"
#include "api/cuda/hpe/Mode.h"
#include "api/cuda/hpe/Module.h"


#ifdef __cplusplus
extern "C" {
#endif

using __impl::core::hpe::KernelDescriptor;

using __impl::cuda::hpe::Mode;
using __impl::cuda::hpe::ModuleDescriptor;
using __impl::cuda::hpe::TextureDescriptor;
using __impl::cuda::hpe::VariableDescriptor;

static inline Mode &getCurrentCUDAMode()
{
    return dynamic_cast<Mode &>(__impl::core::hpe::Thread::getCurrentMode());
}


/*!
 * @returns Module **
 */
GMAC_API void ** APICALL __cudaRegisterFatBinary(void *fatCubin)
{
    TRACE(GLOBAL, "CUDA Fat binary: %p", fatCubin);
    enterGmac();
    ASSERTION(__impl::core::hpe::getProcess().nAccelerators() > 0);
    // Use the first GPU to load the fat binary
    void **ret = (void **) new __impl::cuda::hpe::ModuleDescriptor(fatCubin);
	exitGmac();
	return ret;
}

GMAC_API void APICALL __cudaUnregisterFatBinary(void **fatCubinHandle)
{
    ModuleDescriptor *mod = (ModuleDescriptor *)fatCubinHandle;
    delete mod;
}

GMAC_API void APICALL __cudaRegisterFunction(
		void **fatCubinHandle, const char *hostFun, char * /*devFun*/,
		const char *devName, int /*threadLimit*/, uint3 * /*tid*/, uint3 * /*bid*/,
		dim3 * /*bDim*/, dim3 * /*gDim*/)
{
    TRACE(GLOBAL, "CUDA Function");
	ModuleDescriptor *mod = (ModuleDescriptor *)fatCubinHandle;
	ASSERTION(mod != NULL);
	enterGmac();
    KernelDescriptor k = __impl::core::hpe::KernelDescriptor(devName, (gmac_kernel_id_t) hostFun);
    mod->add(k);
	exitGmac();
}

GMAC_API void APICALL __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
		char * deviceAddress, const char *deviceName, int /*ext*/, int /*size*/,
		int constant, int /*global*/)
{
    TRACE(GLOBAL, "CUDA Variable %s: %p", deviceName, deviceAddress);
	ModuleDescriptor *mod = (ModuleDescriptor *)fatCubinHandle;
	ASSERTION(mod != NULL);
	enterGmac();
    VariableDescriptor v = __impl::cuda::hpe::VariableDescriptor(deviceName, hostVar, bool(constant != 0));
    mod->add(v);
	exitGmac();
}

GMAC_API void APICALL __cudaRegisterTexture(void **fatCubinHandle, const struct textureReference *hostVar,
		const void ** /*deviceAddress*/, const char *deviceName, int /*dim*/, int /*norm*/, int /*ext*/)
{
    TRACE(GLOBAL, "CUDA Texture");
	ModuleDescriptor *mod = (ModuleDescriptor *)fatCubinHandle;
	ASSERTION(mod != NULL);
	enterGmac();
    TextureDescriptor t = __impl::cuda::hpe::TextureDescriptor(deviceName, hostVar);
	mod->add(t);
	exitGmac();
}

GMAC_API void APICALL __cudaRegisterShared(void ** /*fatCubinHandle*/, void ** /*devicePtr*/)
{
}

GMAC_API void APICALL __cudaRegisterSharedVar(void ** /*fatCubinHandle*/, void ** /*devicePtr*/,
		size_t /*size*/, size_t /*alignment*/, int /*storage*/)
{
}

GMAC_API cudaError_t APICALL cudaConfigureCall(dim3 gridDim, dim3 blockDim,
		size_t sharedMem, cudaStream_t tokens)
{
	enterGmac();
    Mode &mode = getCurrentCUDAMode();
	mode.call(gridDim, blockDim, sharedMem, tokens);
	exitGmac();
	return cudaSuccess;
}

GMAC_API cudaError_t APICALL cudaSetupArgument(const void *arg, size_t count, size_t offset)
{
	enterGmac();
    Mode &mode = getCurrentCUDAMode();
	mode.argument(arg, count, (off_t)offset);
	exitGmac();
	return cudaSuccess;
}

GMAC_API gmacError_t APICALL gmacLaunch(const char *k);
                                                    
GMAC_API cudaError_t APICALL cudaLaunch(const char *k)
{
	gmacError_t ret = gmacLaunch(k);
    assert(ret == gmacSuccess);
	return cudaSuccess;
}

#ifdef __cplusplus
}
#endif


