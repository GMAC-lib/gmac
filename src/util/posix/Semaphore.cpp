#include <cassert>

#include "Semaphore.h"

namespace __impl { namespace util {

Semaphore::Semaphore(unsigned v)
{
    pthread_cond_init(&_cond, NULL);
    pthread_mutex_init(&_mutex, NULL);
    _val = v;
}

Semaphore::~Semaphore()
{
    pthread_mutex_destroy(&_mutex);
    pthread_cond_destroy(&_cond);
}

}}
