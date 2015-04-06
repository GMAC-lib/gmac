#include "gtest/gtest.h"

#include "core/hpe/Mode.h"

class IOBufferTest : public testing::Test {
protected:
    static const size_t Size_ = 4 * 1024 * 1024;
	static __impl::core::hpe::Mode *Mode_;
    static gmac::core::hpe::Process *Process_;

	static void SetUpTestCase();
	static void TearDownTestCase();
};

