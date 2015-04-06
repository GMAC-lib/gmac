#ifndef TEST_GMAC_CORE_THREAD_H_
#define TEST_GMAC_CORE_THREAD_H_

#include "core/hpe/Thread.h"

#include "gtest/gtest.h"

class ThreadTest: public testing::Test {
public:
    static __impl::core::hpe::Mode *Mode_;
    static gmac::core::hpe::Process *Process_;
  
	static void SetUpTestCase();
	static void TearDownTestCase();
};


#endif