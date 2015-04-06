#include "unit/core/hpe/Mode.h"

#include "core/hpe/Process.h"
#include "core/hpe/Mode.h"
#include "core/IOBuffer.h"

#include "memory/Memory.h"

#include "gtest/gtest.h"

using gmac::core::hpe::Process;

using __impl::core::hpe::Mode;
using __impl::core::IOBuffer;
using __impl::memory::Object;
using __impl::core::hpe::Accelerator;

Mode *ModeTest::Mode_ = NULL;
Process *ModeTest::Process_ = NULL;

extern void OpenCL(Process &);
extern void CUDA(Process &);

void ModeTest::SetUpTestCase() {
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

void ModeTest::TearDownTestCase() {
    Process_->removeMode(*Mode_);
    Process_->destroy();
    Mode_ = NULL;
    Process_ = NULL;
}

TEST_F(ModeTest, MemoryObject) {
    __impl::memory::ObjectMap &map = Mode_->getAddressSpace();
    Object *obj = map.getProtocol().createObject(*Mode_, Size_, NULL, GMAC_PROT_READ, 0);
    ASSERT_TRUE(obj != NULL);
    obj->addOwner(*Mode_);
    map.addObject(*obj);
    const hostptr_t addr = obj->addr();
    ASSERT_TRUE(addr != 0);
    obj->decRef();

    Object *ref = map.getObject(addr);
    ASSERT_TRUE(ref != NULL);
    ASSERT_EQ(obj, ref);
    map.removeObject(*obj);

    ASSERT_EQ(addr, ref->addr());
    ref->decRef();
}

TEST_F(ModeTest, MemoryObjectMap) {
	__impl::memory::ObjectMap &map = Mode_->getAddressSpace();
	Object *obj = map.getProtocol().createObject(*Mode_, Size_, NULL, GMAC_PROT_WRITE, 0);
	ASSERT_TRUE(obj != NULL);
    obj->addOwner(*Mode_);
	Object *obj2 = map.getProtocol().createObject(*Mode_, Size_, NULL, GMAC_PROT_WRITE, 0);
    ASSERT_TRUE(obj2 != NULL);
    obj2->addOwner(*Mode_);
	ASSERT_FALSE(map.hasObject(*obj));
    ASSERT_TRUE(map.addObject(*obj));
	ASSERT_TRUE(map.hasObject(*obj));

	hostptr_t ptr = obj->addr();
	size_t size_ = obj->size();
    for(size_t s = 0; s < size_; s++) {
       ptr[s] = (s & 0xff);
    }
	size_t size_all = map.memorySize();
	ASSERT_EQ(size_, size_all);
	ASSERT_TRUE(map.addObject(*obj2));
	size_all = map.memorySize();
	ASSERT_EQ(size_*2, size_all);
	ASSERT_TRUE(map.hasModifiedObjects());
	ASSERT_EQ(gmacSuccess, map.releaseObjects());
	ASSERT_TRUE(map.releasedObjects());
	ASSERT_TRUE(map.hasModifiedObjects());
	ASSERT_EQ(gmacSuccess, map.acquireObjects());
	ASSERT_FALSE(map.releasedObjects());

	for(size_t s = 0; s <obj->size(); s++) {
	        EXPECT_EQ(ptr[s], (s & 0xff));
	    }
	ASSERT_TRUE(map.removeObject(*obj));
	ASSERT_TRUE(map.removeObject(*obj2));
	obj->decRef();
	obj2->decRef();
	map.cleanUp();
}

TEST_F(ModeTest, Memory) {
    hostptr_t fakePtr = (uint8_t *) 0xcafebabe;
    accptr_t addr(0);
    ASSERT_EQ(gmacSuccess, Mode_->map(addr, fakePtr, Size_));
    ASSERT_TRUE(addr != 0);
    ASSERT_EQ(gmacSuccess, Mode_->add_mapping(addr, fakePtr, Size_));

    ASSERT_EQ(gmacSuccess, Mode_->unmap(fakePtr, Size_));
}

TEST_F(ModeTest, MemoryCopy) {
    int *src = new int[Size_];
    ASSERT_TRUE(src != NULL);
    int *dst = new int[Size_];
    ASSERT_TRUE(dst != NULL);
    memset(src, 0x5a, Size_ * sizeof(int));
    accptr_t addr(0);
    ASSERT_EQ(gmacSuccess, Mode_->map(addr, hostptr_t(src), Size_ * sizeof(int)));
    ASSERT_TRUE(addr != 0);
    ASSERT_EQ(gmacSuccess, Mode_->add_mapping(addr, hostptr_t(src), Size_ * sizeof(int)));
    ASSERT_EQ(gmacSuccess, Mode_->copyToAccelerator(addr, hostptr_t(src), Size_ * sizeof(int)));
    ASSERT_EQ(gmacSuccess, Mode_->copyToHost(hostptr_t(dst), addr, Size_ * sizeof(int)));
    for(size_t i = 0; i < Size_; i++) ASSERT_EQ(0x5a5a5a5a, dst[i]);
    memset(src, 0x5b, Size_ * sizeof(int));
	accptr_t addr2(0);
	ASSERT_EQ(gmacSuccess, Mode_->map(addr2, hostptr_t(dst), Size_ * sizeof(int)));
    ASSERT_TRUE(addr2 != 0);
    ASSERT_EQ(gmacSuccess, Mode_->add_mapping(addr2, hostptr_t(dst), Size_ * sizeof(int)));
	ASSERT_EQ(gmacSuccess, Mode_->copyToAccelerator(addr, hostptr_t(src), Size_ * sizeof(int)));
	ASSERT_EQ(gmacSuccess, Mode_->copyAccelerator(addr2, addr, Size_ * sizeof(int)));
	ASSERT_EQ(gmacSuccess, Mode_->copyToHost(hostptr_t(dst), addr2, Size_ * sizeof(int)));
    for(size_t i = 0; i < Size_; i++) ASSERT_EQ(0x5b5b5b5b, dst[i]);
    ASSERT_EQ(gmacSuccess, Mode_->unmap(hostptr_t(src), Size_ * sizeof(int)));
	ASSERT_EQ(gmacSuccess, Mode_->unmap(hostptr_t(dst), Size_ * sizeof(int)));
    delete[] dst;
    delete[] src;
}

TEST_F(ModeTest, MemorySet) {
    hostptr_t fakePtr = (uint8_t *) 0xcafebabe;
    accptr_t addr(0);
    ASSERT_EQ(gmacSuccess, Mode_->map(addr, fakePtr, Size_ * sizeof(int)));
    ASSERT_TRUE(addr != 0);
    ASSERT_EQ(gmacSuccess, Mode_->add_mapping(addr, fakePtr, Size_ * sizeof(int)));
    ASSERT_EQ(gmacSuccess, Mode_->memset(addr, 0x5a, Size_ * sizeof(int)));

    int *dst = new int[Size_];
    ASSERT_EQ(gmacSuccess, Mode_->copyToHost(hostptr_t(dst), addr, Size_ * sizeof(int)));
    for(size_t i = 0; i < Size_; i++) ASSERT_EQ(0x5a5a5a5a, dst[i]);

    ASSERT_EQ(gmacSuccess, Mode_->unmap(fakePtr, Size_ * sizeof(int)));
    delete[] dst;
}

TEST_F(ModeTest, IOBuffer) {
    int *mem = new int[Size_];
    ASSERT_TRUE(mem != NULL);
    memset(mem, 0x5a, Size_ * sizeof(int));

    IOBuffer &buffer = Mode_->createIOBuffer(Size_ * sizeof(int), GMAC_PROT_READWRITE);
	ASSERT_EQ(gmacSuccess, Mode_->wait());
    ASSERT_EQ(buffer.addr(), memcpy(buffer.addr(), mem, Size_ * sizeof(int)));
    ASSERT_EQ(0, memcmp(buffer.addr(), mem, Size_ * sizeof(int)));

    accptr_t addr(0);
    hostptr_t fakePtr = (uint8_t *)0xcafecafe;
    ASSERT_EQ(gmacSuccess, Mode_->map(addr, fakePtr, Size_ * sizeof(int)));
    ASSERT_EQ(gmacSuccess, Mode_->add_mapping(addr, fakePtr, Size_ * sizeof(int)));
    ASSERT_EQ(gmacSuccess, Mode_->bufferToAccelerator(addr, buffer, Size_ * sizeof(int)));
    ASSERT_EQ(gmacSuccess, buffer.wait());

    ASSERT_EQ(buffer.addr(), memset(buffer.addr(), 0xa5, Size_ * sizeof(int)));
    ASSERT_EQ(gmacSuccess, Mode_->acceleratorToBuffer(buffer, addr, Size_ * sizeof(int)));
	ASSERT_EQ(gmacSuccess, buffer.wait());
    ASSERT_EQ(0, memcmp(buffer.addr(), mem, Size_ * sizeof(int)));

    ASSERT_EQ(gmacSuccess, Mode_->unmap(fakePtr, Size_ * sizeof(int)));
    Mode_->destroyIOBuffer(buffer);
    delete[] mem;
}

TEST_F(ModeTest, Mode) {
	size_t size=0;
	size_t free=0;
	size_t total=0;
	Mode_->getMemInfo(free, total);
	ASSERT_GE(free, size);
	ASSERT_GE(total, size);
	ASSERT_GE(total, free);

	Accelerator &acc = Mode_->getAccelerator();
	ASSERT_TRUE(&acc != NULL);	
	
	ASSERT_EQ(gmacSuccess, Mode_->prepareForCall());
	Mode *mode = Process_->createMode(ACC_AUTO_BIND);
    ASSERT_TRUE(mode != NULL);
	if(&(mode->getAccelerator()) != &acc) ASSERT_EQ(gmacSuccess, mode->moveTo(acc));
}
