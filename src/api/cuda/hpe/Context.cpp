#include "Context.h"
#include "Mode.h"

#include "hpe/init.h"
#include "memory/Manager.h"
#include "trace/Tracer.h"

namespace __impl { namespace cuda { namespace hpe {

Context::Context(Mode &mode, CUstream streamLaunch, CUstream streamToAccelerator,
                             CUstream streamToHost, CUstream streamAccelerator) :
    gmac::core::hpe::Context(mode, streamLaunch, streamToAccelerator, streamToHost, streamAccelerator),
    call_(dim3(0), dim3(0), 0, NULL, NULL)
{
    call_ = KernelConfig(dim3(0), dim3(0), 0, NULL, streamLaunch_);
}

Context::~Context()
{ 
    // Destroy context's private IOBuffer (if any)
    if(bufferWrite_ != NULL) {
        TRACE(LOCAL,"Destroying I/O buffer");
    	dynamic_cast<Mode &>(mode_).destroyIOBuffer(*bufferWrite_);
    }

    if(bufferRead_ != NULL) {
        TRACE(LOCAL,"Destroying I/O buffer");
    	dynamic_cast<Mode &>(mode_).destroyIOBuffer(*bufferRead_);
    }
}

KernelLaunch &Context::launch(Kernel &kernel)
{
    trace::EnterCurrentFunction();
    KernelLaunch *ret = kernel.launch(dynamic_cast<Mode &>(mode_), call_);
    ASSERTION(ret != NULL);
    trace::ExitCurrentFunction();
    return *ret;
}

}}}
