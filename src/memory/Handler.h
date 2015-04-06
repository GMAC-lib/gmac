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

#ifndef GMAC_MEMORY_HANDLER_H_
#define GMAC_MEMORY_HANDLER_H_

#include <cstdlib>

#include "config/common.h"

namespace __impl {

namespace core { class Process; }

namespace memory {

class Manager;

//! Handler for Read/Write faults
class GMAC_LOCAL Handler {
public:
    typedef void (*CallBack)(void);
private:
    //! Activate the fault handler
	void setHandler();

    //! Deactivate the fault handler
	void restoreHandler(void);

    //! Signal number to bind the handler to
    static int Signum_;

    //! Number of request to activate the handler
	static unsigned Count_;

    //! Active handler
	static Handler *Handler_;
	
    static CallBack Entry_;
    static CallBack Exit_;
public:

    //! Default constructor
	inline Handler() {
		if(Count_ == 0) setHandler();
		Count_++;
	}

    //! Default destructor
	virtual inline ~Handler() { 
		if(--Count_ == 0) restoreHandler();
	}

    //! Set function to be called before executing the handler
    static inline void setEntry(CallBack call) {
        Entry_ = call;
    }

    static inline void Entry() {
        if(Entry_ != NULL) Entry_();
    }

    //! Set function be bo called after executing the handler
    static inline void setExit(CallBack call) {
        Exit_ = call;
    }

    static inline void Exit() {
        if(Exit_ != NULL) Exit_();
    }

    static void setProcess(core::Process &proc);

    static void setManager(Manager &manager);
};

}}
#endif
