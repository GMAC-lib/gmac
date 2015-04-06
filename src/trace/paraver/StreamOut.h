/* Copyright (c) 2011 University of Illinois
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

#ifndef GMAC_TRACE_PARAVER_STREAMOUT_H_
#define GMAC_TRACE_PARAVER_STREAMOUT_H_

#include "Lock.h"

namespace __impl { namespace trace { namespace paraver {

class GMAC_LOCAL StreamOut {
protected:
    paraver::Lock mutex_;
    std::ofstream of_;

public:
    StreamOut(const char *fileName, bool textMode = false)
    {
        std::ios::openmode mode = std::ios::out;
        if(textMode == false) mode |= std::ios::binary;
        of_.open(fileName, mode);
    }

#if 0
    template <typename T>
	friend StreamOut &operator<<(StreamOut &of, const T & t)
    {
        of.mutex_.lock();
        of.of_ << t;
        of.mutex_.unlock();
        return of;
    }
#endif
    // this is the type of std::cout
    typedef std::basic_ostream<char, std::char_traits<char> > CoutType;

    // this is the function signature of std::endl
    typedef CoutType& (*StandardEndLine)(CoutType&);

    StreamOut &operator<<(StandardEndLine t)
    {
        mutex_.lock();
        of_ << t;
        mutex_.unlock();
        return *this;
    }

    template <typename T>
    StreamOut &operator<<(const T &t)
    {
        mutex_.lock();
        of_ << t;
        mutex_.unlock();
        return *this;
    }

	StreamOut &write(const char *s, size_t size)
    {
        mutex_.lock();
        of_.write(s, size);
        mutex_.unlock();
        return *this;
    }

    friend StreamOut &operator<<(StreamOut &of, const char * t);

    void close()
    {
        of_.close();
    }
};

inline
StreamOut &operator<<(StreamOut &of, const char * t)
{
    of.mutex_.lock();
    of.of_ << t;
    of.mutex_.unlock();
    return of;
}



}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
