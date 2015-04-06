/* Copyright (c) 2009, 2011 University of Illinois
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

#ifndef GMAC_UTIL_GMAC_BASE_H_
#define GMAC_UTIL_GMAC_BASE_H_

#include "Atomics.h"
#include "Lock.h"
#include "Logger.h"
#include "Unique.h"

namespace __impl { namespace util {

template <typename T>
class MemoryDebug {
private:
    static Atomic Alloc_;
    static Atomic Free_;

public:
    MemoryDebug()
    {
        unsigned(AtomicInc(Alloc_));
    }

    virtual ~MemoryDebug()
    {
        unsigned(AtomicInc(Free_));
    }

    static void reportDebugInfo()
    {
        printf("Alloc: %u\n", unsigned(Alloc_));
        printf("Free: %u\n", unsigned(Free_));
    }
};

template <typename T>
Atomic MemoryDebug<T>::Alloc_ = 0;

template <typename T>
Atomic MemoryDebug<T>::Free_ = 0;

class Named {
private:
    std::string name_;

public:
    Named(const std::string &name) :
        name_(name)
    {
    }

    std::string &getName()
    {
        return name_;
    }

    const std::string &getName() const
    {
        return name_;
    }
};

typedef void (*report_fn)();

#ifdef DEBUG

class Debug :
    public gmac::util::Lock {
    static Debug debug_;

    typedef std::map<std::string, report_fn> MapTypes;
    MapTypes mapTypes_;

    void registerType_(const std::string &name, report_fn fn)
    {
        lock();
        mapTypes_.insert(std::map<std::string, report_fn>::value_type(name, fn));
        unlock();
    }

    void dumpInfo_()
    {
        if (params::ParamDebugPrintDebugInfo == true) {
            MapTypes::const_iterator it;
            for (it = mapTypes_.begin(); it != mapTypes_.end(); ++it) {
                printf("DEBUG INFORMATION FOR CLASS: %s\n", it->first.c_str());
                it->second();
            }
        }
    }
public:
    Debug() :
        gmac::util::Lock("Debug")
    {
    }

    ~Debug()
    {
        dumpInfo_();
    }

    static void registerType(const std::string &name, report_fn fn)
    {
        debug_.registerType_(name, fn);
    }

    static void dumpInfo()
    {
        debug_.dumpInfo_();
    }
};


#endif

template <typename T>
class GMACBase
#ifdef DEBUG
    :
    public UniqueDebug<T>,
    public MemoryDebug<T>,
    public Named
#endif
{
#ifdef DEBUG
private:
    static std::string getTypeName()
    {
        return std::string(get_name(typeid(T).name()));
    }

    static std::string getTmpName(unsigned id)
    {
        std::stringstream ss;
        ss << std::string(getTypeName()) << "_" << id;
        return ss.str();
    }

    static Atomic registered;

public:
    GMACBase() :
        UniqueDebug<T>(),
        Named(getTmpName(UniqueDebug<T>::getDebugId()))
    {
        if (AtomicInc(registered) == 1) {
            Debug::registerType(getTypeName(), &reportDebugInfo);
        }
    }

    static void
    reportDebugInfo()
    {
        MemoryDebug<T>::reportDebugInfo();
    }

    GMACBase(const std::string &name) :
        Named(name)
    {
    }

    virtual ~GMACBase()
    {
    }
#endif
};

#ifdef DEBUG
template <typename T>
Atomic GMACBase<T>::registered = 0;
#endif

}}


#endif // GMAC_UTIL_GMAC_BASE_H_

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
