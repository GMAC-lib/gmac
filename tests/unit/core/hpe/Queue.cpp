#include "unit/core/hpe/Queue.h"


#include "core/hpe/Process.h"
#include "core/hpe/Mode.h"

#include "gtest/gtest.h"

using __impl::core::hpe::Mode;
using __impl::core::hpe::Queue;
using gmac::core::hpe::Process;
using __impl::core::hpe::ThreadQueue;

Process *QueueTest::Process_ = NULL;

extern void OpenCL(Process &);
extern void CUDA(Process &);

void QueueTest::SetUpTestCase() {
    Process_ = new Process();
    if(Process_ == NULL) return;
#if defined(USE_CUDA)
    CUDA(*Process_);
#endif
#if defined(USE_OPENCL)
    OpenCL(*Process_);
#endif
}

void QueueTest::TearDownTestCase() {
    Process_->destroy();
    Process_ = NULL;
}


TEST_F(QueueTest,PushPop)
{
    ThreadQueue threadQueue; 
    ASSERT_TRUE(threadQueue.queue != NULL);
    Mode *mode = Process_->createMode();
    ASSERT_TRUE(mode != NULL);
    threadQueue.queue->push(mode);
    Mode *last = threadQueue.queue->pop();
    ASSERT_EQ(mode, last);
    Process_->removeMode(*mode);
}





