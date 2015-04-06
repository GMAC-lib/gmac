#ifndef TEST_UNIT_CORE_ADDRESSSPACE_H_
#define TEST_UNIT_CORE_ADDRESSSPACE_H_

#include "gtest/gtest.h"

#include "core/hpe/AddressSpace.h"

class AddressSpaceTest : public testing::Test {
public:
	static __impl::core::hpe::Mode *Mode_;
    static gmac::core::hpe::Process *Process_;

    const static size_t Size_ = 4 * 1024 * 1024;
    
    static void SetUpTestCase();
    static void TearDownTestCase();
};

#endif