#ifndef TEST_GMAC_API_OPENCL_HPE_CONTEXT_H_
#define TEST_GMAC_API_OPENCL_HPE_CONTEXT_H_

#include "gtest/gtest.h"
#include "core/hpe/Process.h"
#include "api/opencl/hpe/ContextFactory.h"

#include "unit/core/hpe/Context.h"

class GMAC_LOCAL OpenCLContextTest :
    public __impl::opencl::hpe::ContextFactory,
    public ContextTest {
public:
	static void SetUpTestCase();
	static void TearDownTestCase();

};

#endif

