#include "api/cuda/hpe/Accelerator.h"
#include "api/cuda/hpe/Kernel.h"
#include "api/cuda/hpe/Mode.h"


namespace __impl { namespace cuda { namespace hpe {

Kernel::Kernel(const core::hpe::KernelDescriptor & k, CUmodule mod) :
    gmac::core::hpe::Kernel(k)
{
    CUresult ret = cuModuleGetFunction(&f_, mod, name_);
    ASSERTION(ret == CUDA_SUCCESS);
    //! \todo Calculate this dynamically
#if CUDA_VERSION >= 3000 && LINUX
    ret = cuFuncSetCacheConfig(f_, CU_FUNC_CACHE_PREFER_L1);
    ASSERTION(ret == CUDA_SUCCESS);
#endif
    ASSERTION(ret == CUDA_SUCCESS);
}

Kernel::~Kernel()
{
    printf("Kernel destroyiiiing\n");
}

KernelConfig::KernelConfig(const KernelConfig & c) :
    argsSize_(0),
    grid_(c.grid_),
    block_(c.block_),
    shared_(c.shared_),
    stream_(c.stream_)
{
    stack_ = new uint8_t[StackSize_];
    ArgsVector::const_iterator it;
    for (it = c.begin(); it != c.end(); ++it) {
        pushArgument(it->ptr(), it->size(), it->offset());
    }
}

KernelConfig::KernelConfig(dim3 grid, dim3 block, size_t shared, cudaStream_t /*tokens*/, CUstream stream) :
    argsSize_(0),
    grid_(grid),
    block_(block),
    shared_(shared),
    stream_(stream)
{
    stack_ = new uint8_t[StackSize_];
}

KernelConfig &KernelConfig::operator=(const KernelConfig &c)
{
    if(&c == this) return *this;
    argsSize_ = 0;
    grid_ = c.grid_;
    block_ = c.block_;
    shared_ = c.shared_;
    stream_ = c.stream_;
    ArgsVector::const_iterator it;
    for (it = c.begin(); it != c.end(); ++it) {
        pushArgument(it->ptr(), it->size(), it->offset());
    }
    return *this;
}

KernelLaunch::KernelLaunch(Mode &mode, const Kernel & k, const KernelConfig & c) :
#ifdef DEBUG
    core::hpe::KernelLaunch(mode, k.key()),
#else
    core::hpe::KernelLaunch(mode),
#endif
    KernelConfig(c),
    kernel_(k),
    f_(k.f_)
{
}

KernelLaunch::~KernelLaunch()
{
    cuEventDestroy(start_);
    cuEventDestroy(end_);
}

gmacError_t
KernelLaunch::execute()
{
	// Set-up parameters
    CUresult ret = cuParamSetv(f_, 0, argsArray(), unsigned(argsSize()));
    CFATAL(ret == CUDA_SUCCESS, "CUDA Error setting parameters: %d", ret);
    ret = cuParamSetSize(f_, unsigned(argsSize()));
	CFATAL(ret == CUDA_SUCCESS);

#if 0
	// Set-up textures
	Textures::const_iterator t;
	for(t = textures_.begin(); t != textures_.end(); t++) {
		cuParamSetTexRef(f_, CU_PARAM_TR_DEFAULT, *(*t));
	}
#endif

	// Set-up shared size
	if((ret = cuFuncSetSharedSize(f_, (unsigned int)shared())) != CUDA_SUCCESS) {
        goto exit;
	}

	if((ret = cuFuncSetBlockShape(f_, block().x, block().y, block().z))
			!= CUDA_SUCCESS) {
        goto exit;
	}

    if((ret = cuEventCreate(&start_, CU_EVENT_DEFAULT)) != CUDA_SUCCESS) {
        goto exit;
    }

    if((ret = cuEventCreate(&end_, CU_EVENT_DEFAULT)) != CUDA_SUCCESS) {
        goto exit;
    }

    if((ret = cuEventRecord(start_, stream_)) != CUDA_SUCCESS) {
        goto exit;
    }

	if((ret = cuLaunchGridAsync(f_, grid().x, grid().y, stream_)) != CUDA_SUCCESS) {
        goto exit;
    }

    cuEventRecord(end_, stream_);

exit:
    return Accelerator::error(ret);
}

}}}
