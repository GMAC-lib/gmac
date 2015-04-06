#include "Context.h"
#include "core/hpe/Process.h"
#include "api/opencl/hpe/Context.h"
#include "api/opencl/hpe/Mode.h"

using gmac::core::hpe::Process;
using __impl::opencl::hpe::Mode;
using __impl::opencl::hpe::Context;
using __impl::opencl::hpe::ContextFactory;

extern void OpenCL(Process &);

void OpenCLContextTest::SetUpTestCase() {
    Process_ = new Process();
#if defined(USE_OPENCL)
    OpenCL(*Process_);
#endif
}

void OpenCLContextTest::TearDownTestCase() {
    Process_->destroy();
    Process_ = NULL;
}


/* We need to use this ugly hack because GTF will declare class
 * methods with default visibility
 */
#if defined(__GNUC__)
#pragma GCC visibility push(hidden)
#endif

TEST_F(OpenCLContextTest, ContextMemory)
{
    unsigned count = unsigned(Process_->nAccelerators());
    for(unsigned i = 0; i < count; i++) {
        Mode *mode = dynamic_cast<Mode *>(Process_->createMode(i));
        ASSERT_TRUE(mode != NULL);

        cl_command_queue queue = mode->getAccelerator().createCLstream();
        Context *ctx = ContextFactory::create(*mode, queue);
        ASSERT_TRUE(ctx != NULL);

        ContextTest::Memory(*mode, *ctx);

        ContextFactory::destroy(*ctx);
        mode->getAccelerator().destroyCLstream(queue);
        Process_->removeMode(*mode);
    }
}

#if defined(__GNUC__)
#pragma GCC visibility pop
#endif

