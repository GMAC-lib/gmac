#include "Record.h"
#include "Element.h"

#include <typeinfo>

namespace __impl { namespace trace { namespace paraver {

Record *Record::read(std::ifstream &in)
{
	Type type;
	in.read((char *)&type, sizeof(type));
	assert(in.eof() == false);
	switch(type) {
		case STATE:
			return new State(in);
		case EVENT:
			return new Event(in);
        case COMM:
            return new Communication(in);
		case LAST:
			return NULL;
		default:
			abort();
	};
	return NULL;
}

StreamOut & operator<<(StreamOut &os, const Record &record)
{
	if(typeid(record) == typeid(State)) 
		os << dynamic_cast<const State &>(record);
	else if(typeid(record) == typeid(Event)) 
		os << dynamic_cast<const Event &>(record);
    else if(typeid(record) == typeid(Communication))
        os << dynamic_cast<const Communication &>(record);
	
	return os;
}


State::State(Thread *thread) :
	id_(thread->getTask(), thread->getApp(), thread->getId()),
	start_(0),
	end_(0),
	state_(-1)
{ }

Event::Event(Thread *thread, uint64_t when, uint64_t event, int64_t value) :
	id_(thread->getTask(), thread->getApp(), thread->getId()),
	when_(when),
	event_(event),
	value_(value)
{ }

Communication::Communication(Thread *src, Thread *dst, uint64_t start, uint64_t end, uint64_t size) :
	src_(src->getTask(), src->getApp(), src->getId()),
	dst_(dst->getTask(), dst->getApp(), dst->getId()),
    start_(start), end_(end), size_(size)
{
}



} } }
