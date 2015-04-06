#include <ctime>
#include <cstdlib>
#include <list>

#include "gtest/gtest.h"

#include "util/allocator/Buddy.h"

using gmac::util::allocator::Buddy;

class BuddyTest : public testing::Test {
public:
    static const size_t Size_ = 8 * 1024 * 1024;
    static long_t Base_;
    static void *BasePtr_;

    static void SetUpTestCase() {
    }

    static void TearDownTestCase() {
    }
};


long_t BuddyTest::Base_ = 0x1000;
void *BuddyTest::BasePtr_ = (void *)Base_;

TEST_F(BuddyTest, SimpleAllocations) {
    Buddy buddy(hostptr_t(BasePtr_), Size_);
    ASSERT_TRUE(buddy.addr() == BasePtr_);

    std::list<hostptr_t> allocs;
    hostptr_t ret = 0;
    for(unsigned n = 2; n < 32; n = n * 2) {
        size_t size = Size_ / n;
        ret = buddy.get(size);
        ASSERT_EQ(Base_, long_t(ret) % (Size_ / n));
        ASSERT_EQ(size, (Size_ / n));
        allocs.push_back(ret);
    }

    size_t size = Size_ / 2;
    ret = buddy.get(size);
    ASSERT_TRUE(ret == NULL);
    ASSERT_EQ(size, Size_ / 2);

    std::list<hostptr_t>::const_iterator i;
    unsigned n = 2;
    for(i = allocs.begin(); i != allocs.end(); i++) {
        buddy.put(*i, Size_ / n);
        n = n * 2;
    }
    allocs.clear();

    size = Size_ / 2;
    ret = buddy.get(size);
    ASSERT_TRUE(ret != NULL);
    ASSERT_EQ(size, Size_ / 2);
}

TEST_F(BuddyTest, RandomAllocations) {
    const int Allocations = 1024;
    const size_t MinMemory = 4 * 4096;
    size_t freeMemory = Size_;
    Buddy buddy(hostptr_t(BasePtr_), Size_);
    ASSERT_TRUE(buddy.addr() == BasePtr_);

    typedef std::map<hostptr_t, size_t> AllocMap;
    AllocMap map;
    srand(unsigned(time(NULL)));
    for(int i = 0; i < Allocations; i++) {
        // Generate a random size to be allocated
        size_t s = 0;
        while(s == 0) s = size_t(freeMemory * rand() / (RAND_MAX + 1.0));

        // Perform the allocation
        size_t check = s;
        hostptr_t addr = buddy.get(s);
        if(addr == NULL) {--i; continue; } // Too much memory, try again
        ASSERT_GE(s, check);
        std::pair<AllocMap::iterator, bool> p = map.insert(AllocMap::value_type(addr, s));
        ASSERT_EQ(true, p.second);
        freeMemory -= s;

        if(freeMemory > MinMemory) continue;

        int n = int(map.size() * rand() / (RAND_MAX + 1.0));
        AllocMap::iterator b = map.begin();
        for(; n > 0; n--, b++);
        buddy.put(b->first, b->second);
        freeMemory += b->second;
        map.erase(b);
    }

}
