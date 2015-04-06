#include "core/hpe/Process.h"
#include "core/hpe/Mode.h"
#include "unit/core/hpe/Thread.h" 

#include "memory/Memory.h"

#include "gtest/gtest.h"

using gmac::core::hpe::Process;

using __impl::core::hpe::Mode;
using __impl::core::hpe::Thread;

Mode *ThreadTest::Mode_ = NULL;
Process *ThreadTest::Process_ = NULL;

extern void OpenCL(Process &);
extern void CUDA(Process &);

void ThreadTest::SetUpTestCase() {
   Process_ = new Process();
    if(Process_ == NULL) return;
#if defined(USE_CUDA)
    CUDA(*Process_);
#endif
#if defined(USE_OPENCL)
    OpenCL(*Process_);
#endif
    __impl::memory::Init();
    ASSERT_TRUE(Mode_ == NULL);
    Mode_ = Process_->createMode(0);
    ASSERT_TRUE(Mode_ != NULL);

}

void ThreadTest::TearDownTestCase() {
    Process_->removeMode(*Mode_);
    Process_->destroy();
    Mode_ = NULL;
    Process_ = NULL;

}

TEST_F(ThreadTest, ThreadMode) {
	//Thread &thread = Thread(*Process_);
	//ASSERT_TRUE(&thread != NULL);  // Constructure can not work

	ASSERT_FALSE(Thread::hasCurrentMode());
	Thread::setCurrentMode(Mode_);
	
	ASSERT_EQ(Mode_, &Thread::getCurrentMode());
    Thread::setCurrentMode(NULL);  // Remove mode	
}
