#if defined(USE_TRACE_PARAVER)

#include "Paraver.h"

#include "util/Parameter.h"
#include "paraver/Pcf.h"

namespace __impl { namespace trace {

void InitApiTracer()
{
	tracer = new __impl::trace::Paraver();
}
void FiniApiTracer()
{
	if(tracer != NULL) delete tracer;
}

Paraver::Paraver() :
    baseName_(std::string(util::params::ParamTrace)),
    fileName_(baseName_ + ".trace"),
    trace_(fileName_.c_str(), 1, trace::GetThreadId()),
    enabled_(true)
{
    FunctionEvent_ = paraver::Factory<paraver::EventName>::create("Function");

#ifdef USE_TRACE_LOCKS
    LockEventRequest_ = paraver::Factory<paraver::EventName>::create("LockRequest");
    LockEventAcquireExclusive_ = paraver::Factory<paraver::EventName>::create("LockAcquireExclusive");
    LockEventAcquireShared_ = paraver::Factory<paraver::EventName>::create("LockAcquireShared");
#endif

#   define STATE(s) \
        states_.insert(StateMap::value_type(s, paraver::Factory<paraver::StateName>::create(EnumState<s>::name())));
#   include "States-def.h"
#   undef STATE
}

Paraver::~Paraver()
{
    enabled_ = false;
    trace_.write(timeMark());

    paraver::TraceReader reader(fileName_.c_str());
    std::string prvFile = baseName_ + ".prv";
    paraver::StreamOut prv(prvFile.c_str(), true);
    prv << reader;
    prv.close();

    std::string pcfFile = baseName_ + ".pcf";
    std::ofstream pcf(pcfFile.c_str());
    paraver::pcf(pcf);
    pcf.close();
}

void Paraver::startThread(uint64_t t, THREAD_T tid, const char *name)
{
    if(enabled_ == false) return;
    trace_.addThread(1, tid);
}

void Paraver::endThread(uint64_t t, THREAD_T tid)
{
}

void Paraver::enterFunction(uint64_t t, THREAD_T tid, const char *name)
{
    if(enabled_ == false) return;
    int32_t id = 0;
    mutex_.lock();
    FunctionMap::const_iterator i = functions_.find(std::string(name));
    if(i == functions_.end()) {
        id = int32_t(functions_.size() + 1);
        FunctionEvent_->registerType(id, std::string(name));
        functions_.insert(FunctionMap::value_type(std::string(name), id));
    }
    else id = i->second;
    mutex_.unlock();
    trace_.pushEvent(t, 1, tid, *FunctionEvent_, id);
}

void Paraver::exitFunction(uint64_t t, THREAD_T tid, const char *name)
{
    if(enabled_ == false) return;
    trace_.pushEvent(t, 1, tid, *FunctionEvent_, 0);
}

#ifdef USE_TRACE_LOCKS
void Paraver::requestLock(uint64_t t, THREAD_T tid, const char *name)
{
    if(enabled_ == false) return;
    int32_t id = 0;
    mutex_.lock();
    LockMap::const_iterator i = locksRequest_.find(std::string(name));
    if(i == locksRequest_.end()) {
        id = int32_t(locksRequest_.size() + 1);
        LockEventRequest_->registerType(id, std::string(name));
        locksRequest_.insert(LockMap::value_type(std::string(name), id));
    }
    else id = i->second;
    mutex_.unlock();
    trace_.pushEvent(t, 1, tid, *LockEventRequest_, id);
}

void Paraver::acquireLockExclusive(uint64_t t, THREAD_T tid, const char *name)
{
    if(enabled_ == false) return;
    int32_t id = 0;
    trace_.pushEvent(t, 1, tid, *LockEventRequest_, 0);
    mutex_.lock();
    LockMap::const_iterator i = locksExclusive_.find(std::string(name));
    if(i == locksExclusive_.end()) {
        id = int32_t(locksExclusive_.size() + 1);
        LockEventAcquireExclusive_->registerType(id, std::string(name));
        locksExclusive_.insert(LockMap::value_type(std::string(name), id));
    }
    else id = i->second;
    mutex_.unlock();
    trace_.pushEvent(t, 1, tid, *LockEventAcquireExclusive_, id);
}

void Paraver::acquireLockShared(uint64_t t, THREAD_T tid, const char *name)
{
    if(enabled_ == false) return;
    int32_t id = 0;
    trace_.pushEvent(t, 1, tid, *LockEventRequest_, 0);
    mutex_.lock();
    LockMap::const_iterator i = locksShared_.find(std::string(name));
    if(i == locksShared_.end()) {
        id = int32_t(locksShared_.size() + 1);
        LockEventAcquireShared_->registerType(id, std::string(name));
        locksShared_.insert(LockMap::value_type(std::string(name), id));
    }
    else id = i->second;
    mutex_.unlock();
    trace_.pushEvent(t, 1, tid, *LockEventAcquireShared_, id);
}

void Paraver::exitLock(uint64_t t, THREAD_T tid, const char *name)
{
    if(enabled_ == false) return;
    mutex_.lock();
    LockMap::const_iterator i = locksExclusive_.find(std::string(name));
    if(i == locksExclusive_.end()) {
        trace_.pushEvent(t, 1, tid, *LockEventAcquireExclusive_, 0);
    } else if(locksShared_.find(std::string(name)) != locksShared_.end()) {
        trace_.pushEvent(t, 1, tid, *LockEventAcquireShared_, 0);
    }
    mutex_.unlock();
}
#endif

void Paraver::setThreadState(uint64_t t, THREAD_T tid, const State state)
{
    if(enabled_ == false) return;
    trace_.pushState(t, 1, tid, *states_[state]);
}

void Paraver::dataCommunication(uint64_t t, THREAD_T src, THREAD_T dst, uint64_t delta, size_t size)
{
    if(enabled_ == false) return;
    trace_.pushCommunication(t, 1, src, t + delta, 1, dst, size);
}

}}

#endif
