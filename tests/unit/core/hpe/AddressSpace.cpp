#include "unit/core/hpe/AddressSpace.h"

#include "core/hpe/Process.h"
#include "core/hpe/Mode.h"
#include "core/hpe/AddressSpace.h"

#include "gtest/gtest.h"

using gmac::core::hpe::Process;

using __impl::core::hpe::Mode;
using __impl::memory::Object;
using __impl::core::hpe::AddressSpace;

Mode *AddressSpaceTest::Mode_ = NULL;
Process *AddressSpaceTest::Process_ = NULL;

extern void OpenCL(Process &);
extern void CUDA(Process &);

void AddressSpaceTest::SetUpTestCase() {
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

void AddressSpaceTest::TearDownTestCase() {
    Process_->removeMode(*Mode_);
    Process_->destroy();
    Mode_ = NULL;
    Process_ = NULL;
}

TEST_F(AddressSpaceTest, Creation) {
	AddressSpace *aSpace = new AddressSpace("TestASpace", *Process_);
	ASSERT_TRUE(aSpace != NULL);
}

TEST_F(AddressSpaceTest, Object) {
	__impl::memory::ObjectMap &map = Mode_->getAddressSpace();
	ASSERT_TRUE(&map != NULL);
	__impl::memory::Protocol &proto = map.getProtocol();
	ASSERT_TRUE(&proto != NULL);
	Object *object = proto.createObject(*Mode_, Size_, NULL, GMAC_PROT_READ, 0);
	ASSERT_TRUE(object != NULL);
    object->addOwner(*Mode_);
	
	hostptr_t ptr = object->addr();
	ASSERT_TRUE(ptr != NULL);
	AddressSpace *aSpace = new AddressSpace("TestASpace", *Process_);
	ASSERT_TRUE(aSpace != NULL);
	ASSERT_TRUE(aSpace->addObject(*object));
	ASSERT_TRUE(aSpace->hasObject(*object));
	ASSERT_EQ(object, aSpace->getObject(ptr, Size_));
    
	ASSERT_TRUE(aSpace->removeObject(*object));
	ASSERT_FALSE(aSpace->hasObject(*object));
	ASSERT_NE(object, aSpace->getObject(ptr, Size_)); 

    object->removeOwner(*Mode_);
}
