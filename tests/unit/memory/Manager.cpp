#include "gtest/gtest.h"

#include "core/IOBuffer.h"
#include "core/hpe/Mode.h"
#include "core/hpe/Process.h"
#include "core/hpe/Thread.h"
#include "memory/Manager.h"
#include "memory/ObjectMap.h"
#include "memory/Object.h"

using namespace gmac::core::hpe;
using namespace gmac::memory;

using __impl::memory::ObjectMap;
 
class ManagerTest : public testing::Test {
public:
    static Process *Process_;
    static const size_t Size_;

    static void SetUpTestCase();
    static void TearDownTestCase();
};

Process *ManagerTest::Process_ = NULL;
const size_t ManagerTest::Size_ = 4 * 1024 * 1024;

extern void OpenCL(Process &);
extern void CUDA(Process &);

void ManagerTest::SetUpTestCase()
{
    Process_ = new Process();
    ASSERT_TRUE(Process_ != NULL);
#if defined(USE_CUDA)
    CUDA(*Process_);
#endif
#if defined(USE_OPENCL)
    OpenCL(*Process_);
#endif
}

void ManagerTest::TearDownTestCase()
{
    ASSERT_TRUE(Process_ != NULL);
    Process_->destroy(); Process_ = NULL;
}

TEST_F(ManagerTest, Creation) {
	ASSERT_TRUE(Process_ != NULL);
	for(int i = 0; i < 16; i++) {
		Manager *manager = new Manager(*Process_);
		ASSERT_TRUE(manager != NULL);
        manager->destroy();
	}
}

TEST_F(ManagerTest, Alloc) {
	ASSERT_TRUE(Process_ != NULL);
    Manager *manager = new Manager(*Process_);
    ASSERT_TRUE(manager != NULL);

	for(size_t size = 4096; size < Size_; size *= 2) {
		hostptr_t ptr = NULL;
		size_t size_ = 0;
		ASSERT_EQ(gmacSuccess, manager->alloc(Thread::getCurrentMode(), &ptr, size));
        ASSERT_TRUE(ptr != NULL);

		ASSERT_EQ(gmacSuccess, manager->getAllocSize(Thread::getCurrentMode(), ptr, size_));
		ASSERT_EQ(size, size_);

		ASSERT_EQ(gmacSuccess, manager->memset(Thread::getCurrentMode(), ptr, 0x5a, size));
		for(size_t i = 0; i < size; i++) {
			ASSERT_TRUE(ptr[i]==0x5a);
		}
     
        ASSERT_EQ(gmacSuccess, manager->free(Thread::getCurrentMode(), ptr));
	}
    manager->destroy();
}

TEST_F(ManagerTest, GlobalAllocReplicated) {
	ASSERT_TRUE(Process_ != NULL);
    Manager *manager = new Manager(*Process_);
    ASSERT_TRUE(manager != NULL);
    
	for(size_t size = 4096; size < Size_; size *= 2) {
		hostptr_t ptr = NULL;
        ASSERT_EQ(gmacSuccess, manager->globalAlloc(Thread::getCurrentMode(), &ptr, size,
                                                    GMAC_GLOBAL_MALLOC_REPLICATED));
        ASSERT_TRUE(ptr != NULL);

        ASSERT_EQ(gmacSuccess, manager->free(Thread::getCurrentMode(), ptr));
	}
    manager->destroy();
}

#if !defined(USE_OPENCL)
TEST_F(ManagerTest, GlobalAllocCentralized) {
	ASSERT_TRUE(Process_ != NULL);
    Manager *manager = new Manager(*Process_);
    ASSERT_TRUE(manager != NULL);
    
    for(size_t size = 4096; size < Size_; size *= 2) {
        hostptr_t ptr = NULL;
        ASSERT_EQ(gmacSuccess, manager->globalAlloc(Thread::getCurrentMode(), &ptr, size,
                                                    GMAC_GLOBAL_MALLOC_CENTRALIZED));
        ASSERT_TRUE(ptr != NULL);

        ASSERT_EQ(gmacSuccess, manager->free(Thread::getCurrentMode(), ptr));
    }
    manager->destroy();
}
#endif

TEST_F(ManagerTest, Coherence) {
	ASSERT_TRUE(Process_ != NULL);
    Manager *manager = new Manager(*Process_);
    ASSERT_TRUE(manager != NULL);

    hostptr_t ptr = NULL;
    ASSERT_EQ(gmacSuccess, manager->alloc(Thread::getCurrentMode(), &ptr, Size_));
    ASSERT_TRUE(ptr != NULL);
    ASSERT_TRUE(manager->translate(Thread::getCurrentMode(), ptr).get() != NULL);

    ObjectMap &map = Thread::getCurrentMode().getAddressSpace();
	ASSERT_TRUE(map.hasModifiedObjects());

    for(int n = 0; n < 16; n++) {

	    for(size_t s = 0; s < Size_; s++) {
	        ptr[s] = (s & 0xff);
	    }
        ASSERT_TRUE(map.hasModifiedObjects());
	
    	ASSERT_EQ(gmacSuccess, manager->releaseObjects(Thread::getCurrentMode()));
        ASSERT_TRUE(map.releasedObjects());
        
    	ASSERT_EQ(gmacSuccess, manager->acquireObjects(Thread::getCurrentMode()));
        ASSERT_FALSE(map.releasedObjects());
	    ASSERT_FALSE(map.hasModifiedObjects());

	    for(size_t s = 0; s < Size_; s++) {
	        EXPECT_EQ(ptr[s], (s & 0xff));
	    }
        ASSERT_FALSE(map.hasModifiedObjects());

        for(size_t s = 0; s < Size_; s++) {
	        ptr[s] = 0x0;
	    }
        ASSERT_TRUE(map.hasModifiedObjects());
    }

    ASSERT_EQ(gmacSuccess, manager->free(Thread::getCurrentMode(), ptr));
    manager->destroy();
}

TEST_F(ManagerTest, IOBufferWrite) {
    ASSERT_TRUE(Process_ != NULL);
    Manager *manager = new Manager(*Process_);
    ASSERT_TRUE(manager != NULL);

    hostptr_t ptr = NULL;
    ASSERT_EQ(gmacSuccess, manager->alloc(Thread::getCurrentMode(), &ptr, Size_));
    ASSERT_TRUE(ptr != NULL);
    ASSERT_TRUE(manager->translate(Thread::getCurrentMode(), ptr).get() != NULL);

    __impl::core::IOBuffer &buffer = Thread::getCurrentMode().createIOBuffer(Size_, GMAC_PROT_READWRITE);

    for(size_t n = 0; n < 16; n++) {
	    for(size_t s = 0; s < Size_; s++) {
	        ptr[s] = (s & 0xff);
	    }

        memset(buffer.addr(), 0x5a, Size_);
        ASSERT_EQ(gmacSuccess, manager->fromIOBuffer(Thread::getCurrentMode(), ptr + n * 128,
                                                     buffer, n * 128, Size_ - n * 128));

        for(size_t s = 0; s < n * 128; s++) EXPECT_EQ(ptr[s], (s & 0xff));
	    for(size_t s = n * 128; s < Size_; s++) EXPECT_EQ(ptr[s], 0x5a);
    }
    Thread::getCurrentMode().destroyIOBuffer(buffer);
    ASSERT_EQ(gmacSuccess, manager->free(Thread::getCurrentMode(), ptr));
    manager->destroy();
}

TEST_F(ManagerTest, IOBufferRead) {
    ASSERT_TRUE(Process_ != NULL);
    Manager *manager = new Manager(*Process_);
    ASSERT_TRUE(manager != NULL);

    hostptr_t ptr = NULL;
    ASSERT_EQ(gmacSuccess, manager->alloc(Thread::getCurrentMode(), &ptr, Size_));
    ASSERT_TRUE(ptr != NULL);
    ASSERT_TRUE(manager->translate(Thread::getCurrentMode(), ptr).get() != NULL);

    __impl::core::IOBuffer &buffer = Thread::getCurrentMode().createIOBuffer(Size_, GMAC_PROT_READWRITE);

    for(size_t n = 0; n < 16; n++) {
	    for(size_t s = n * 128; s < Size_; s++) {
	        ptr[s] = (s & 0xff);
	    }

        memset(buffer.addr(), 0x5a, Size_);
        ASSERT_EQ(gmacSuccess, manager->toIOBuffer(Thread::getCurrentMode(), buffer, n * 128,
                                                     ptr + n * 128, Size_ - n * 128));

        for(size_t s = 0; s < n * 128; s++) EXPECT_EQ(buffer.addr()[s], 0x5a);
	    for(size_t s = n * 128; s < Size_; s++) EXPECT_EQ(buffer.addr()[s], (s & 0xff));
    }
    Thread::getCurrentMode().destroyIOBuffer(buffer);
    ASSERT_EQ(gmacSuccess, manager->free(Thread::getCurrentMode(), ptr));
    manager->destroy();
}

TEST_F(ManagerTest, Memcpy) {
	ASSERT_TRUE(Process_ != NULL);
	Manager *manager = new Manager(*Process_);
	ASSERT_TRUE(manager != NULL);

	for(size_t size = 4096;size < Size_;size *= 2) {
		hostptr_t ptr_src = NULL;
		hostptr_t ptr_dst = NULL;
		
		ASSERT_EQ(gmacSuccess, manager->alloc(Thread::getCurrentMode(), &ptr_src, size));
		ASSERT_TRUE(ptr_src != NULL);
		ASSERT_EQ(gmacSuccess, manager->alloc(Thread::getCurrentMode(), &ptr_dst, size));
		ASSERT_TRUE(ptr_dst != NULL);
		
		ASSERT_EQ(gmacSuccess, manager->memset(Thread::getCurrentMode(), ptr_src, 0x5a, size));
		ASSERT_EQ(gmacSuccess, manager->memset(Thread::getCurrentMode(), ptr_dst, 0xaa, size));
		ASSERT_EQ(gmacSuccess, manager->memcpy(Thread::getCurrentMode(), ptr_dst, ptr_src, size));
		
		for(size_t i = 0; i < size; i++) {
			ASSERT_TRUE(ptr_src[i] == ptr_dst[i]);
			ASSERT_TRUE(ptr_dst[i] == 0x5a);
			
		}
		ASSERT_EQ(gmacSuccess, manager->free(Thread::getCurrentMode(), ptr_src));
		ASSERT_EQ(gmacSuccess, manager->free(Thread::getCurrentMode(), ptr_dst));
	}

	ASSERT_EQ(gmacSuccess, manager->flushDirty(Thread::getCurrentMode()));
	manager->destroy();
}
