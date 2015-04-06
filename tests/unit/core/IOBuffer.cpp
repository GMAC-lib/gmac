#include "unit/core/IOBuffer.h"

#include "core/IOBuffer.h"
#include "core/hpe/Process.h"
#include "core/hpe/Mode.h"

#include "gtest/gtest.h"

using gmac::core::hpe::Process;

using __impl::core::hpe::Mode;
using __impl::core::IOBuffer;

Mode *IOBufferTest::Mode_ = NULL;
Process *IOBufferTest::Process_ = NULL;

extern void OpenCL(Process &);
extern void CUDA(Process &);

void IOBufferTest::SetUpTestCase() {
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

void IOBufferTest::TearDownTestCase() {
    Process_->removeMode(*Mode_);
    Process_->destroy();
    Mode_ = NULL;
    Process_ = NULL;
}
TEST_F(IOBufferTest, ToAccelerator) {
	IOBuffer &buffer = Mode_->createIOBuffer(Size_, GMAC_PROT_WRITE);
    ASSERT_TRUE(buffer.size() >= Size_);

    ASSERT_TRUE(memset(buffer.addr(), 0x7a, buffer.size()) == buffer.addr());

    hostptr_t fakePtr = (uint8_t *) 0xcafebabe;
    accptr_t addr(0);
    ASSERT_EQ(gmacSuccess, Mode_->map(addr, fakePtr, Size_));
    ASSERT_EQ(gmacSuccess, Mode_->add_mapping(addr, fakePtr, Size_));

    ASSERT_EQ(gmacSuccess, Mode_->bufferToAccelerator(addr, buffer, Size_));

    ASSERT_EQ(gmacSuccess, buffer.wait());
    ASSERT_EQ(IOBuffer::Idle, buffer.state());

    int *dst = NULL;
    dst = new int[Size_ / sizeof(int)];
    ASSERT_TRUE(dst != NULL);
    ASSERT_EQ(gmacSuccess, Mode_->copyToHost(hostptr_t(dst), addr, Size_));
    for(size_t i = 0; i < Size_ / sizeof(int); i++) ASSERT_EQ(0x7a7a7a7a, dst[i]);

    ASSERT_EQ(gmacSuccess, Mode_->unmap(fakePtr, Size_));
    delete[] dst;

    Mode_->destroyIOBuffer(buffer);
}
TEST_F(IOBufferTest, ToHost) {
	IOBuffer &buffer = Mode_->createIOBuffer(Size_, GMAC_PROT_READ);
    ASSERT_TRUE(buffer.size() >= Size_);
	ASSERT_EQ(GMAC_PROT_READ, buffer.getProtection());
    ASSERT_EQ(buffer.addr() + Size_, buffer.end()); 
	ASSERT_TRUE(buffer.async());

    hostptr_t fakePtr = (uint8_t *) 0xcafebabe;
    accptr_t addr(0);
    ASSERT_EQ(gmacSuccess, Mode_->map(addr, fakePtr, Size_));
    ASSERT_EQ(gmacSuccess, Mode_->add_mapping(addr, fakePtr, Size_));
    ASSERT_EQ(gmacSuccess, Mode_->memset(addr, 0x5b, Size_));
    
    ASSERT_EQ(gmacSuccess, Mode_->acceleratorToBuffer(buffer, addr, Size_));

    ASSERT_EQ(gmacSuccess, buffer.wait());
    ASSERT_EQ(IOBuffer::Idle, buffer.state());

    int *ptr = reinterpret_cast<int *>(buffer.addr());
    for(size_t i = 0; i < Size_ / sizeof(int); i++) ASSERT_EQ(0x5b5b5b5b, ptr[i]);
    ASSERT_EQ(gmacSuccess, Mode_->unmap(fakePtr, Size_));

    Mode_->destroyIOBuffer(buffer);
}
