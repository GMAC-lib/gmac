#include "gtest/gtest.h"

#include "core/IOBuffer.h"
#include "core/Mode.h"
#include "core/hpe/Mode.h"
#include "core/hpe/Process.h"
#include "core/hpe/Thread.h"
#include "memory/Manager.h"
#include "memory/Object.h"

using gmac::core::hpe::Thread;

using __impl::core::Mode;
using __impl::memory::Object;

extern void OpenCL(gmac::core::hpe::Process &);
extern void CUDA(gmac::core::hpe::Process &);

class ObjectTest : public testing::Test {
protected:
    static gmac::core::hpe::Process *Process_;
    static gmac::memory::Manager *Manager_;
        static const size_t Size_;

        static void SetUpTestCase();
        static void TearDownTestCase();
};


gmac::core::hpe::Process *ObjectTest::Process_ = NULL;
gmac::memory::Manager *ObjectTest::Manager_ = NULL;
const size_t ObjectTest::Size_ = 4 * 1024 * 1024;

void ObjectTest::SetUpTestCase()
{
    Process_ = new gmac::core::hpe::Process();
    ASSERT_TRUE(Process_ != NULL);
#if defined(USE_CUDA)
    CUDA(*Process_);
#endif
#if defined(USE_OPENCL)
    OpenCL(*Process_);
#endif
    Manager_ = new gmac::memory::Manager(*Process_);
}

void ObjectTest::TearDownTestCase()
{
    ASSERT_TRUE(Manager_ != NULL);
    Manager_->destroy();
    Manager_ = NULL;

    ASSERT_TRUE(Process_ != NULL);
    Process_->destroy();
    Process_ = NULL;
}

TEST_F(ObjectTest, Creation)
{
    ASSERT_TRUE(Process_ != NULL);
    Mode &mode = Thread::getCurrentMode();
    __impl::memory::ObjectMap &map = mode.getAddressSpace();
    __impl::memory::Protocol &proto = map.getProtocol();
    Object *object = proto.createObject(mode, Size_, NULL, GMAC_PROT_READ, 0);
    ASSERT_TRUE(object != NULL);

    map.removeObject(*object);
    object->decRef();
}

TEST_F(ObjectTest, Blocks)
{
    ASSERT_TRUE(Process_ != NULL);
    Mode &mode = Thread::getCurrentMode();
    __impl::memory::ObjectMap &map = mode.getAddressSpace();
    Object *object = map.getProtocol().createObject(mode, Size_, NULL, GMAC_PROT_READ, 0);
    ASSERT_TRUE(object != NULL);
    object->addOwner(mode);
    hostptr_t start = object->addr();
    ASSERT_TRUE(start != NULL);
    hostptr_t end = object->end();
    ASSERT_TRUE(end != NULL);
    size_t blockSize = object->blockSize();
    ASSERT_GT(blockSize, 0U);

    for(size_t offset = 0; offset < object->size(); offset += blockSize) {
        EXPECT_EQ(0, object->blockBase(offset));
        EXPECT_EQ(blockSize, object->blockEnd(offset));
    }

    map.removeObject(*object);
    object->decRef();
}

TEST_F(ObjectTest, Coherence)
{
    ASSERT_TRUE(Process_ != NULL);
    Mode &mode = Thread::getCurrentMode();
    __impl::memory::ObjectMap &map = mode.getAddressSpace();
    Object *object = map.getProtocol().createObject(mode, Size_, NULL, GMAC_PROT_READ, 0);
    ASSERT_TRUE(object != NULL);
    object->addOwner(mode);
    ASSERT_TRUE(object->addr() != NULL);
    ASSERT_TRUE(object->end() != NULL);
    ASSERT_EQ(Size_, size_t(object->end() - object->addr()));
    ASSERT_EQ(Size_, object->size());
    map.addObject(*object);

    hostptr_t ptr = object->addr();
    for(size_t s = 0; s < object->size(); s++) {
       ptr[s] = (s & 0xff);
    }
    ASSERT_EQ(gmacSuccess, object->release());
    ASSERT_EQ(gmacSuccess, object->toAccelerator());

    GmacProtection prot = GMAC_PROT_READWRITE;
    ASSERT_EQ(gmacSuccess, object->acquire(prot));
    mode.memset(object->acceleratorAddr(mode, object->addr()), 0, Size_);

    for(size_t s = 0; s < object->size(); s++) {
        EXPECT_EQ(ptr[s], 0);
    }

    map.removeObject(*object);
    object->decRef();
}

TEST_F(ObjectTest, IOBuffer)
{
    ASSERT_TRUE(Process_ != NULL);
    Mode &mode = Thread::getCurrentMode();
    __impl::memory::ObjectMap &map = mode.getAddressSpace();
    Object *object = map.getProtocol().createObject(mode, Size_, NULL, GMAC_PROT_READ, 0);
    ASSERT_TRUE(object != NULL);
    object->addOwner(mode);
    map.addObject(*object);

    __impl::core::IOBuffer &buffer = mode.createIOBuffer(Size_, GMAC_PROT_READWRITE);

    hostptr_t ptr = buffer.addr();
    for(size_t s = 0; s < buffer.size(); s++) {
        ptr[s] = (s & 0xff);
    }

    ASSERT_EQ(gmacSuccess, object->copyFromBuffer(buffer, Size_));

    ptr = buffer.addr();
    memset(ptr, 0, Size_);

    ASSERT_EQ(gmacSuccess, object->copyToBuffer(buffer, Size_));
    ASSERT_EQ(gmacSuccess, buffer.wait());

    ptr = buffer.addr();
    int error = 0;
    for(size_t s = 0; s < buffer.size(); s++) {
        //EXPECT_EQ(ptr[s], (s & 0xff));
        error += (ptr[s] - (s & 0xff));
    }
    EXPECT_EQ(error, 0);

    mode.destroyIOBuffer(buffer);

    map.removeObject(*object);
    object->decRef();
}
