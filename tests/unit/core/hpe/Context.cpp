#include "Context.h"
#include "core/hpe/Context.h"
#include "core/hpe/Mode.h"

using gmac::core::hpe::Process;
using __impl::core::hpe::Mode;
using gmac::core::hpe::Context;

Process *ContextTest::Process_ = NULL;

void ContextTest::Memory(Mode &mode, Context &ctx)
{
	int *buffer  = new int[Size_];
	int *canary  = new int[Size_];

    accptr_t device(0);
    ASSERT_TRUE(mode.map(device, hostptr_t(buffer), Size_ * sizeof(int)) == gmacSuccess); 
    ASSERT_TRUE(device != 0);

    memset(buffer, 0xa5, Size_ * sizeof(int));
    memset(canary, 0x5a, Size_ * sizeof(int));

    ASSERT_TRUE(ctx.copyToAccelerator(device, hostptr_t(buffer), Size_ * sizeof(int)) == gmacSuccess);
    ASSERT_TRUE(memcmp(buffer, canary, Size_ * sizeof(int)) != 0);
    ASSERT_TRUE(ctx.copyToHost(hostptr_t(canary), device, Size_ * sizeof(int)) == gmacSuccess);
    ASSERT_TRUE(memcmp(buffer, canary, Size_ * sizeof(int)) == 0);

    memset(buffer,  0, Size_ * sizeof(int));
    memset(canary,  0, Size_ * sizeof(int));
    ASSERT_TRUE(memcmp(buffer, canary, Size_ * sizeof(int)) == 0);

    ASSERT_TRUE(ctx.copyToHost(hostptr_t(buffer), device, Size_ * sizeof(int)) == gmacSuccess);
    ASSERT_TRUE(memcmp(buffer, canary, Size_ * sizeof(int)) != 0);

    ASSERT_TRUE(mode.unmap(hostptr_t(buffer),  Size_ * sizeof(int)) == gmacSuccess);

	delete[] canary;
	delete[] buffer;
}


