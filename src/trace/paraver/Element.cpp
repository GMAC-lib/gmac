#include "Element.h"

namespace __impl { namespace trace { namespace paraver {

void Thread::start(StreamOut &of, unsigned s, uint64_t t)
{ 
	// Flush the previous state to disk (if needed)
    if(current_ != NULL) {
        if(current_->getId() != s) {
	        current_->end(t);
    	    current_->write(of);
            delete current_;
        }
        else return;
    }

	// Setup the new state
    current_ = new State(this);
	current_->start(s, t);
}

void Thread::end(StreamOut &of, uint64_t t)
{
    if(current_ == NULL) return;
	// Flush previous state to disk (if needed)
	current_->end(t);
    current_->write(of);
    delete current_;
    current_ = NULL;
}

StreamOut &operator<<(StreamOut &os, const Application &app) 
{
	std::map<int32_t, Task *>::const_iterator i;
	os << app.sons_.size();
	os << "(";
	for(i = app.sons_.begin(); i != app.sons_.end(); ++i) {
		if(i != app.sons_.begin()) os << ",";
		os << i->second->size() << ":" << 0;
	}
	os << ")";
	return os;
}

} } }
