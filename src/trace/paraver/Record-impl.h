#ifndef GMAC_TRACE_PARAVER_RECORD_IMPL_H_
#define GMAC_TRACE_PARAVER_RECORD_IMPL_H_

namespace __impl { namespace trace { namespace paraver {

inline
void Record::end(StreamOut &of)
{
    Type type = LAST;
    of.write((const char *)&type, sizeof(type));
}


inline
bool RecordPredicate::operator()(const Record *a, const Record *b)
{
    if(a->getTime() < b->getTime()) return true;
	else if(a->getTime() > b->getTime()) return false;
	else {
        if(a->getType() < b->getType()) return true;
		else if(a->getType() > b->getType()) return false;
		else if(a->getId() < b->getId()) return true;
    }
    return false;
}

inline
RecordId::RecordId(int32_t task, int32_t app, int32_t thread) :
    task_(task), app_(app), thread_(thread)
{
}

inline
RecordId::RecordId(std::ifstream &in)
{
    in.read((char *)&task_, sizeof(task_));
    in.read((char *)&app_, sizeof(app_));
    in.read((char *)&thread_, sizeof(thread_));
}
	
inline
void RecordId::write(StreamOut &of) const
{
    of.write((const char *)&task_, sizeof(task_));
    of.write((const char *)&app_, sizeof(app_));
    of.write((const char *)&thread_, sizeof(thread_));
}

inline
StreamOut & operator<<(StreamOut &os, const RecordId &id)
{
    os << 0 << ":" << id.task_ << ":" << id.app_ << ":" << id.thread_;
    return os;
}

inline
State::State(std::ifstream &in) :
    id_(in)
{
    in.read((char *)&start_, sizeof(start_));
    in.read((char *)&end_, sizeof(end_));
    in.read((char *)&state_, sizeof(state_));

}

inline
int State::getType() const
{
    return STATE;
}

inline
uint64_t State::getTime() const
{
    return start_;
}

inline
uint64_t State::getEndTime() const
{
    return end_;
}

inline
uint32_t State::getId() const
{
    return state_;
}

inline
void State::start(uint32_t state, uint64_t start)
{ 
    state_ = state;
    start_ = start;
}

inline
void State::restart(uint64_t start)
{
    start_ = start;
}

inline
void State::end(uint64_t end)
{
    end_ = end;
}

inline
void State::write(StreamOut &of) const
{
    if(start_ >= end_) return;
    Type type = STATE;
    of.write((const char *)&type, sizeof(type));
    id_.write(of);
    of.write((const char *)&start_, sizeof(start_));
    of.write((const char *)&end_, sizeof(end_));
    of.write((const char *)&state_, sizeof(state_));
}

inline
StreamOut & operator<<(StreamOut &os, const State &state)
{
    os << Record::STATE << ":" << state.id_ << ":" << state.start_ << ":";
    os << state.end_ << ":" << state.state_ << std::endl;
    return os;
}

inline
Event::Event(std::ifstream &in) :
    id_(in)
{
    in.read((char *)&when_, sizeof(when_));
    in.read((char *)&event_, sizeof(event_));
    in.read((char *)&value_, sizeof(value_));
}

inline
int Event::getType() const
{
    return EVENT;
}

inline
uint64_t Event::getTime() const
{
    return when_;
}

inline
uint64_t Event::getEndTime() const
{
    return when_;
}

inline
uint32_t Event::getId() const
{
    return uint32_t(event_);
}

inline
void Event::write(StreamOut &of) const
{
    Type type = EVENT;
    of.write((const char *)&type, sizeof(type));
    id_.write(of);
    of.write((const char *)&when_, sizeof(when_));
    of.write((const char *)&event_, sizeof(event_));
    of.write((const char *)&value_, sizeof(value_));
}

inline
StreamOut & operator<<(StreamOut &os, const Event &event)
{
    os << Record::EVENT << ":" << event.id_ << ":" << event.when_ << ":";
    os << event.event_ << ":" << event.value_ << std::endl;
    return os;
}


inline
Communication::Communication(std::ifstream &in) :
    src_(in), dst_(in)
{
    in.read((char *)&start_, sizeof(start_));
    in.read((char *)&end_, sizeof(end_));
    in.read((char *)&size_, sizeof(size_));
}

inline
int Communication::getType() const
{
    return COMM;
}

inline
uint64_t Communication::getTime() const
{
    return start_;
}

inline
uint64_t Communication::getEndTime() const
{
    return end_;
}

inline
uint32_t Communication::getId() const
{
    return uint32_t(size_);
}

inline
void Communication::write(StreamOut &of) const
{
    Type type = COMM;
    of.write((const char *)&type, sizeof(type));
    src_.write(of);
    dst_.write(of);
    of.write((const char *)&start_, sizeof(start_));
    of.write((const char *)&end_, sizeof(end_));
    of.write((const char *)&size_, sizeof(size_));

}

inline
StreamOut & operator<<(StreamOut &os, const Communication &comm)
{
    static int id = 1;
    os << Record::COMM << ":" << comm.src_ << ":" << comm.start_ << ":" << comm.start_ << ":";
    os << comm.dst_ << ":" << comm.end_ << ":" << comm.end_ << ":" << comm.size_ << ":";
    os << id << std::endl;
    return os;
}


} } }

#endif
