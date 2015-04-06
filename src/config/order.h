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

#ifndef GMAC_CONFIG_ORDER_H_
#define GMAC_CONFIG_ORDER_H_

#include "common.h"

/* Common constructors */
namespace __impl { namespace util { namespace params {
void GMAC_API Init(void);
}}}

/* Interposition Constructors */
void osInit(void) GMAC_LOCAL;
void threadInit(void) GMAC_LOCAL;
void stdcInit(void) GMAC_LOCAL;
#ifdef USE_MPI
void mpiInit(void) GMAC_LOCAL;
#endif

#ifdef __cplusplus
namespace __impl { 

namespace memory {
class Protocol;

void Init(void);
#define GLOBAL_PROTOCOL 0x1
Protocol *ProtocolInit(unsigned flags);
void Fini(void);
}

namespace core {
class Accelerator;
class IOBuffer;
class Mode;
class Context;
class Process;

void apiInit(void);
void contextInit(void);

}}
#endif
#endif
