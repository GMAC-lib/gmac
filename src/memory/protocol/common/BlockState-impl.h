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

#ifndef GMAC_MEMORY_PROTOCOL_COMMON_BLOCKSTATE_IMPL_H_
#define GMAC_MEMORY_PROTOCOL_COMMON_BLOCKSTATE_IMPL_H_

namespace __impl {
namespace memory { namespace protocol { namespace common {

template <typename T>
inline
BlockState<T>::BlockState(ProtocolState state) :
    state_(state),
    faultsCacheWrite_(0),
    faultsCacheRead_(0)
{
}

template <typename T>
inline
T
BlockState<T>::getState() const
{
    return state_;
}

template <typename T>
inline
unsigned
BlockState<T>::getCacheWriteFaults() const
{
    return faultsCacheWrite_;
}

template <typename T>
inline
unsigned
BlockState<T>::getCacheReadFaults() const
{
    return faultsCacheRead_;
}

template <typename T>
inline
void
BlockState<T>::resetCacheWriteFaults()
{
    faultsCacheWrite_ = 0;
}

template <typename T>
inline
void
BlockState<T>::resetCacheReadFaults()
{
    faultsCacheRead_ = 0;
}

#if 0
template <typename T>
inline
void
BlockState<T>::setState(ProtocolState state)
{
    state_ = state;
}
#endif

}}}}

#endif /* BLOCKSTATE_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
