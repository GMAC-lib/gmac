#include "Names.h"

namespace __impl { namespace trace { namespace paraver {

template<> int32_t Factory<StateName>::next_ = 0;
template<> int32_t Factory<EventName>::next_ = 0;

template<> Factory<StateName>::List *Factory<StateName>::items_ = NULL;
template<> Factory<EventName>::List *Factory<EventName>::items_ = NULL;

} } }
