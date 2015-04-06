#ifndef TEST_GMAC_CORE_QUEUE_H_
#define TEST_GMAC_CORE_QUEUE_H_

#include "gtest/gtest.h"
#include "core/hpe/Process.h"

class QueueTest:public testing::Test {

public:
    static gmac::core::hpe::Process *Process_;

    static void SetUpTestCase();
    static void TearDownTestCase();

};
#endif
