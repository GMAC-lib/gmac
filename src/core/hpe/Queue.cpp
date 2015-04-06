#include "Queue.h"
#include "util/Logger.h"

namespace __impl { namespace core { namespace hpe {

Queue::Queue(const char *name) :
    gmac::util::Lock(name), sem(0)
{}

void Queue::push(Mode *mode)
{
    lock();
    _queue.push_back(mode);
    unlock();
    sem.post();
}

Mode * Queue::pop()
{
    sem.wait();
    lock();
    ASSERTION(_queue.empty() == false);
    Mode *ret = _queue.front();
    _queue.pop_front();
    unlock();
    return ret;
}

ThreadQueue::ThreadQueue()
{
    queue = new Queue("ThreadQueue");
}

ThreadQueue::~ThreadQueue()
{
    delete queue;
}

}}}
