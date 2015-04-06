/* Copyright (c) 2009 University of Illinois
                   Universitat Politecnica de Catalunya
                   All rights reserved.

Developed by: IMPACT Research Group / Grup de Sistemes Operatius
              University of Illinois / Universitat Politecnica de Catalunya
              http://impact.crhc.illinois.edu/
              http://gso.ac.upc.edu/

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal with the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
  1. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimers.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimers in the
     documentation and/or other materials provided with the distribution.
  3. Neither the names of IMPACT Research Group, Grup de Sistemes Operatius,
     University of Illinois, Universitat Politecnica de Catalunya, nor the
     names of its contributors may be used to endorse or promote products
     derived from this Software without specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
WITH THE SOFTWARE.  */

#include "gtest/gtest.h"
#include "core/AllocationMap.h"

class AllocationMapTest : public testing::Test {
public:
#if defined(USE_OPENCL)
    static cl_context Context_;
    static void SetUpTestCase();
    static void TearDownTestCase();

    static cl_mem Allocate(size_t size = 4096);
    static void Release(cl_mem mem);
#endif

};

#if defined(USE_OPENCL)
cl_context AllocationMapTest::Context_ = NULL;

void AllocationMapTest::SetUpTestCase()
{
    cl_platform_id platform;
    cl_device_id device;
    cl_int error_code = CL_SUCCESS;
    
    ASSERT_EQ(CL_SUCCESS, clGetPlatformIDs(1, &platform, NULL));
    ASSERT_EQ(CL_SUCCESS, clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL));
    Context_ = clCreateContext(NULL, 1, &device, NULL, NULL, &error_code);
    ASSERT_EQ(CL_SUCCESS, error_code);
}

void AllocationMapTest::TearDownTestCase()
{
    if(Context_ != NULL) clReleaseContext(Context_);
    Context_ = NULL;
}

cl_mem AllocationMapTest::Allocate(size_t size)
{
    cl_mem ret = NULL;
    cl_int error_code = CL_SUCCESS;
    if(Context_ == NULL) return cl_mem(NULL);
    ret = clCreateBuffer(Context_, CL_MEM_READ_WRITE, size, NULL, &error_code);
    if(error_code != CL_SUCCESS) return cl_mem(NULL);
    return ret;
}

void AllocationMapTest::Release(cl_mem mem)
{
    if(mem != NULL) clReleaseMemObject(mem);
}
#endif

TEST_F(AllocationMapTest, Insertion)
{
    gmac::core::AllocationMap map_;
    hostptr_t host((hostptr_t)0xcafecafe);
#if defined(USE_CUDA)
    accptr_t device((accptr_t)0xcacacaca);
#elif defined(USE_OPENCL)
    accptr_t device(Allocate());
    ASSERT_NE(device, 0);
#endif
    size_t size = 1024;
    map_.insert(host, device, size);

    accptr_t retDevice(0);
    size_t retSize;
    ASSERT_TRUE(map_.find(host, retDevice, retSize));
    ASSERT_TRUE(device == retDevice);
    ASSERT_EQ(size, retSize);
    
    map_.erase(host, size);
    ASSERT_FALSE(map_.find(host, retDevice, retSize));
#if defined(USE_OPENCL)
    Release(device.get());
#endif
}
