#ifndef USE_MULTI_CONTEXT

#include "core/hpe/Accelerator.h"
#include "core/hpe/AddressSpace.h"
#include "core/hpe/Process.h"
#include "core/hpe/Thread.h"
#include "core/hpe/Mode.h"

#include "Accelerator.h"

using gmac::core::hpe::Process;
using gmac::core::hpe::Thread;

using __impl::core::hpe::Accelerator;
using __impl::core::hpe::AddressSpace;
using __impl::core::hpe::Mode;

Process *AcceleratorTest::Process_ = NULL;
std::vector<Accelerator *> Accelerators_;

extern void OpenCL(Process &);
extern void CUDA(Process &);

void AcceleratorTest::SetUpTestCase()
{
    Process_ = new Process();
#if defined(USE_CUDA)
    CUDA(*Process_);
#endif
#if defined(USE_OPENCL)
    OpenCL(*Process_);
#endif
}

void AcceleratorTest::TearDownTestCase()
{
    if(Process_ != NULL) Process_->destroy();
    Process_ = NULL;
}

TEST_F(AcceleratorTest, Memory) {
    int *buffer = new int[Size_];
    int *canary = new int[Size_];

    memset(buffer, 0xa5, Size_ * sizeof(int));
    memset(canary, 0x5a, Size_ * sizeof(int));
    accptr_t device(0);
    size_t count = Process_->nAccelerators();
    for(unsigned n = 0; n < count; n++) {
        Accelerator &acc = Process_->getAccelerator(n);
		ASSERT_EQ(n, acc.id());
        ASSERT_TRUE(acc.map(device, hostptr_t(buffer), Size_ * sizeof(int)) == gmacSuccess);
        ASSERT_TRUE(device != 0);
        ASSERT_TRUE(acc.add_mapping(device, hostptr_t(buffer), Size_ * sizeof(int)) == gmacSuccess);
		ASSERT_TRUE(acc.getMapping(device, hostptr_t(buffer), Size_ * sizeof(int)));
		
		ASSERT_TRUE(acc.copyToAccelerator(device, hostptr_t(buffer), Size_ * sizeof(int), Thread::getCurrentMode()) == gmacSuccess);
        ASSERT_TRUE(acc.copyToHost(hostptr_t(canary), device, Size_ * sizeof(int), Thread::getCurrentMode()) == gmacSuccess);
        ASSERT_TRUE(memcmp(buffer, canary, Size_ * sizeof(int)) == 0);  //compare mem size
        ASSERT_TRUE(acc.unmap(hostptr_t(buffer), Size_ * sizeof(int)) == gmacSuccess);
    }
    delete[] canary;
    delete[] buffer;
}

TEST_F(AcceleratorTest, Aligment) {
    const hostptr_t fakePtr = (uint8_t *) 0xcafebabe;
    const int max = 32 * 1024 * 1024;
    size_t count = Process_->nAccelerators();
    for(unsigned i = 0; i < count; i++) {
        Accelerator &acc = Process_->getAccelerator(i);
        for(int n = 1; n < max; n <<= 1) {
            accptr_t device(0);
            ASSERT_TRUE(acc.map(device, fakePtr, Size_, n) == gmacSuccess);
            ASSERT_TRUE(device != 0);
            ASSERT_TRUE(acc.add_mapping(device, fakePtr, Size_) == gmacSuccess);
            ASSERT_TRUE(acc.unmap(fakePtr, Size_) == gmacSuccess);
        }
    }

}

TEST_F(AcceleratorTest, CreateMode) {
	size_t size = 0;
	size_t free = 0;
	size_t total = 0;
	size_t n = Process_->nAccelerators();
	std::vector<AddressSpace *> aSpaces;
    for(unsigned i = 0; i < n; i++) {
        Accelerator &acc = Process_->getAccelerator(i);
		acc.getMemInfo(free, total);
		ASSERT_GE(free, size);
		ASSERT_GE(total, size);
		ASSERT_GE(total, free);
        unsigned load = acc.load();
		AddressSpace *aSpace = new AddressSpace("TestASpace", *Process_);
        aSpaces.push_back(aSpace);
        Mode *mode = acc.createMode(*Process_, *aSpace);
        ASSERT_TRUE(mode != NULL);
        ASSERT_TRUE(acc.load() == load + 1);

        mode->decRef();
        ASSERT_TRUE(acc.load() == load);
    }

    for(unsigned i = 0; i < n; i++) {
        delete aSpaces[i];
    }
}

#endif


