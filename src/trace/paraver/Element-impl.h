#ifndef GMAC_TRACE_PARAVER_ELEMENT_IMPL_H_
#define GMAC_TRACE_PARAMER_ELEMENT_IMPL_H_

#include <sstream>

namespace __impl { namespace trace { namespace paraver {

inline
Abstract::Abstract(int32_t id, std::string name) :
    id_(id), name_(name)
{}

inline
int32_t Abstract::getId() const
{
    return id_;
}

inline
std::string Abstract::getName() const
{
    std::ostringstream os;
    os << name_ << id_;
    return os.str();
}


template<typename P, typename S>
inline
P *Element<P, S>::getParent() const
{
    return parent_;
}

template<typename P, typename S>
inline
void Element<P, S>::addSon(int32_t id, S *son)
{
    sons_[id] = son;
}

template<typename P, typename S>
inline
S *Element<P, S>::getSon(int32_t id) const
{
    S * ret = NULL;
    typename std::map<int32_t, S *>::const_iterator i;
    i = sons_.find(id);
    if(i != sons_.end()) ret = i->second;
    return ret;
}

template<typename P, typename S>
inline
Element<P, S>::Element(P * parent, int32_t id, std::string name) :
    Abstract(id, name),
    parent_(parent)
{ }

template<typename P, typename S>
inline
Element<P, S>::~Element()
{
    typename std::map<int32_t, S *>::const_iterator i;
    for(i = sons_.begin(); i != sons_.end(); i++)
        if(i->second) delete i->second;
}

template<typename P, typename S>
inline
size_t Element<P, S>::size() const
{
    size_t ret = sons_.size();
    return ret;
}

template<typename P, typename S>
inline
void Element<P, S>::end(StreamOut &os, uint64_t t) const
{
    typename std::map<int32_t, S *>::const_iterator i;
    for(i = sons_.begin(); i != sons_.end(); i++)
        if(i->second != NULL) i->second->end(os, t);
}

template<typename P, typename S>
inline
void Element<P, S>::write(StreamOut &os) const
{
    uint32_t s = uint32_t(sons_.size());
    os.write((const char *)&id_, sizeof(id_));
    os.write((const char *)&s, sizeof(s));
    typename std::map<int32_t, S *>::const_iterator i;
    for(i = sons_.begin(); i != sons_.end(); i++)
        i->second->write(os);
}

template<typename P>
inline
P *Element<P, void>::getParent() const
{
    return parent_;
}

template<typename P>
inline
Element<P, void>::Element(P *parent, int32_t id, std::string name) :
    Abstract(id, name),
    parent_(parent)
{ }

template<typename P>
inline
Element<P, void>::~Element()
{
}

template<typename P>
inline
size_t Element<P, void>::size() const 
{
    return 0;
}

template<typename P>
inline
void Element<P, void>::end(StreamOut &os, uint64_t t) const
{
}

template<typename P>
inline
void Element<P, void>::write(StreamOut &of) const
{
    of.write((const char *)&id_, sizeof(id_));
    uint32_t s = uint32_t(size());
    of.write((const char *)&s, sizeof(s));
}


inline
Thread::Thread(Task *task, int32_t id, int32_t tid) :
    Element<Task, void>(task, id, "Thread"),
    current_(NULL),
    tid_(tid)
{
    current_ = new State(this);
    current_->start(State::None, 0);
}

inline
int32_t Thread::getTask() const
{
    return parent_->getId();
}

inline
int32_t Thread::getApp() const
{
    return parent_->getApp();
}

inline 
int32_t Thread::getTid() const
{
    return tid_;
}

inline
bool Thread::ready() const
{
    return current_ != NULL;
}


inline
Task::Task(Application *app, int32_t id) :
    Element<Application, Thread>(app, id, "Task"),
    threads_(1)
{
}

inline
Thread *Task::addThread(int32_t id)
{
    Thread *thread = new Thread(this, threads_++, id);
    addSon(id, thread);
    return thread;
}

inline
Thread *Task::getThread(int32_t id) const
{
    return getSon(id);
}

inline int32_t Task::getApp() const
{
    return parent_->getId();
}


inline
Application::Application(int32_t id, std::string name) :
    Element<void, Task>(NULL, id, name),
    tasks_(1)
{
}

inline
Task *Application::getTask(int32_t id) const
{
    return getSon(id);
}

inline
Task *Application::addTask(int32_t id)
{
    Task *task = new Task(this, tasks_++);
    addSon(id, task);
    return task;
}

} } }

#endif
