#ifndef GMAC_TRACE_PARAVER_NAMES_IMPL_H_
#define GMAC_TRACE_PARAVER_NAMES_IMPL_H_

namespace __impl { namespace trace { namespace paraver {

template<typename T>
inline
void Factory<T>::init()
{
    if(items_ != NULL) return;
    next_ = 0;
    items_ = new List();
}

template<typename T>
inline
T *Factory<T>::create(const char *name)
{
    init();
    T *ret = new T(name, next_++);
    items_->push_back(ret);
    return ret;
}

template<typename T>
inline
bool Factory<T>::valid()
{
    return items_ != NULL;
}

template<typename T>
inline
const typename Factory<T>::List &Factory<T>::get()
{
    ASSERTION(items_ != NULL);
    return *items_;
}

template<typename T>
inline
void Factory<T>::destroy()
{
    if(items_ == NULL) return;
    typename List::const_iterator i;
    for(i = items_->begin(); i != items_->end(); i++)
        delete *i;
    delete items_;
}


inline
Name::Name(const char *name, int32_t value) :
    name_(name), value_(value)
{
}

inline
std::string Name::getName() const
{
    return name_;
}

inline
int32_t Name::getValue() const
{
    return value_;
}


inline
StateName::StateName(const char *name, int32_t value) : 
    Name(name, value)
{ }


inline
EventName::EventName(const char *name, int32_t value) :
    Name(name, value)
{ }

inline
void EventName::registerType(uint32_t value, std::string type)
{
    types_.insert(TypeTable::value_type(value, type));
}

inline
const EventName::TypeTable &EventName::getTypes() const
{
    return types_;
}

} } }

#endif
