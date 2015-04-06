#include "Semaphore.h"
#include <cassert>

namespace __impl { namespace util {

Semaphore::Semaphore(unsigned v)
{
    InitializeConditionVariable(&cond_);
    InitializeCriticalSection(&mutex_);
    val_ = v;
}

Semaphore::~Semaphore()
{
    DeleteCriticalSection(&mutex_);
}

}}
