#include "api/cuda/hpe/Context.h"
#include "api/cuda/hpe/ContextFactory.h"

namespace __impl { namespace cuda { namespace hpe {

Context *
ContextFactory::create(Mode &mode, CUstream streamLaunch, CUstream streamToAccelerator,
                            CUstream streamToHost, CUstream streamAccelerator) const
{
    return new Context(mode, streamLaunch, streamToAccelerator,
                             streamToHost, streamAccelerator);
}

void ContextFactory::destroy(Context &context) const
{
    delete &context;
}

}}}
