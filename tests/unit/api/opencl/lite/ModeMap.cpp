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
#include "api/opencl/lite/ModeMap.h"
#include "OpenCL.h"

#include <list>

class ModeMapTest : public testing::Test {
protected:
    static const size_t num_modes = 8;

    static void CreateContexts(std::list<cl_context> &contexts, gmac::opencl::lite::ModeMap &map, size_t num_modes);
    static void CleanUpContexts(std::list<cl_context> &contexts, gmac::opencl::lite::ModeMap &map);
};


void ModeMapTest::CreateContexts(std::list<cl_context> &contexts, gmac::opencl::lite::ModeMap &map, size_t num_modes)
{
    for(unsigned n = 0; n < num_modes; n++) {
        cl_context context;
        cl_device_id device;
        ASSERT_TRUE(CreateOpenCLContext(device, context));
        contexts.push_back(context);
        map.insert(context, *(new gmac::opencl::lite::Mode(context, 1, &device)));
    }
}

void ModeMapTest::CleanUpContexts(std::list<cl_context> &contexts, gmac::opencl::lite::ModeMap &map)
{
    while(contexts.empty() != false) {
        ASSERT_TRUE(map.get(contexts.front()) != NULL);
        gmac::opencl::lite::ModeMap::iterator it;
        it = map.find(contexts.front());
        ASSERT_TRUE(it != map.end());
        map.erase(it);
        ASSERT_TRUE(map.get(contexts.front()) == NULL);
        ASSERT_EQ(CL_SUCCESS, clReleaseContext(contexts.front()));
        contexts.pop_front();
    }
}


TEST_F(ModeMapTest, Insertion)
{
    gmac::opencl::lite::ModeMap map; 
    std::list<cl_context> contexts;

    SCOPED_TRACE("CreateContext");
    CreateContexts(contexts, map, num_modes);
    if(HasFatalFailure()) return;
    SCOPED_TRACE("CleanUpContext");
    CleanUpContexts(contexts, map);
    if(HasFatalFailure()) return;
}

TEST_F(ModeMapTest, Removal)
{
    unsigned n;
    gmac::opencl::lite::ModeMap map; 
    std::list<cl_context> contexts;

    SCOPED_TRACE("CreateContext");
    CreateContexts(contexts, map, num_modes);
    if(HasFatalFailure()) return;
        
    std::list<cl_context>::iterator i;
    for(n = 0, i = contexts.begin(); i != contexts.end(); n++) {
        if((n % 2) == 0) { i++; continue; }
        ASSERT_TRUE(map.get(*i) != NULL);
        map.remove(*i);
        ASSERT_TRUE(map.get(*i) == NULL);
        ASSERT_EQ(CL_SUCCESS, clReleaseContext(*i));
        i = contexts.erase(i);
    }

    for(i = contexts.begin(); i != contexts.end(); i++) {
        ASSERT_TRUE(map.get(*i) != NULL);
    }

    SCOPED_TRACE("CleanUpContext");
    CleanUpContexts(contexts, map);
    if(HasFatalFailure()) return;
}

