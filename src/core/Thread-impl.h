/*
 * <+ DESCRIPTION +>
 *
 * Copyright (C) 2011, Javier Cabezas <jcabezas in ac upc edu> {{{
 *
 * This program is free software; you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License 
 * as published by the Free Software Foundation; either 
 * version 2 of the License, or any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 * }}}
 */

#ifndef GMAC_CORE_THREAD_IMPL_H_
#define GMAC_CORE_THREAD_IMPL_H_

#include "util/Logger.h"
#include "util/Private.h"

namespace __impl { namespace core {

inline void
TLS::Init()
{
    __impl::util::Private<Thread>::init(CurrentThread_);
}

inline
Thread &
TLS::getCurrentThread()
{
    ASSERTION(TLS::CurrentThread_.get() != NULL);
    return *TLS::CurrentThread_.get();
}

inline
Thread::Thread() :
    lastError_(gmacSuccess)
{
    ASSERTION(TLS::CurrentThread_.get() == NULL);
    TLS::CurrentThread_.set(this);
#ifdef DEBUG
    debugTID_ = THREAD_T(AtomicInc(Thread::NextTID_));
#endif
}

inline
Thread::~Thread()
{
    TLS::CurrentThread_.set(NULL);
}

#ifdef DEBUG
inline
THREAD_T
Thread::getDebugTID()
{
    if (TLS::CurrentThread_.get() == NULL) return (THREAD_T)1;
    return TLS::getCurrentThread().debugTID_;
}
#endif


inline
gmacError_t &
Thread::getLastError()
{
    return TLS::getCurrentThread().lastError_;
}

inline
void
Thread::setLastError(gmacError_t error)
{
    TLS::getCurrentThread().lastError_ = error;
}

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
