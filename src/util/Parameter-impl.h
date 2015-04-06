#ifndef GMAC_UTIL_PARAMETER_IMPL_H_
#define GMAC_UTIL_PARAMETER_IMPL_H_

#include <cstdio>
#include <cstdlib>

#if defined(__GNUC__)
#	define GETENV getenv
#elif defined(_MSC_VER)
#	define GETENV gmac_getenv
static inline const char *gmac_getenv(const char *name)
{
	static char buffer[512];
	size_t size = 0;
	if(getenv_s(&size, buffer, 512, name) != 0) return NULL;
	if(strlen(buffer) == 0) return NULL;
	return (const char *)buffer;
}
#endif

namespace __impl { namespace util {

template <typename T>
static
T convert(const char * str);

template <>
inline bool convert<bool>(const char * str)
{
    return bool(atoi(str) != 0);
}

template <>
inline int convert<int>(const char * str)
{
    return atoi(str);
}

template <>
inline unsigned convert<unsigned>(const char * str)
{
    return unsigned(atoi(str));
}

template<>
inline long_t convert<long_t>(const char *str)
{
    return long_t(atoi(str));
}

template <>
inline float convert<float>(const char * str)
{
    return (float)atof(str);
}

template <>
inline char * convert<char *>(const char * str)
{
    return (char *)str;
}

template <>
inline const char * convert<const char *>(const char * str)
{
    return str;
}

template<typename T>
inline Parameter<T>::Parameter(T *value, const char *name,
        T def, const char *envVar, uint32_t flags) :
    value_(value),
    def_(def),
    name_(name),
    envVar_(envVar),
    flags_(flags)
{
    const char *tmp = NULL;
    if(envVar_ != NULL &&
        (tmp = GETENV(envVar_)) != NULL) {
        *value_ = convert<T>(tmp);

        if (flags_ & PARAM_NONZERO && *value_ == 0) {
            *value_ = def_;
        } else {
            envSet_ = true;
        }
    }
    else {
        *value_ = def_;
    }
    
}

template<typename T>
void Parameter<T>::print() const
{
    std::cout << name_ << std::endl;
    std::cout << "\tValue: " << *value_ << std::endl;
    std::cout << "\tDefault: " << def_ << std::endl;
    std::cout << "\tVariable: " << envVar_ << std::endl;
    std::cout << "\tFlags: " << flags_ << std::endl;
    std::cout << "\tSet: " << envSet_ << std::endl;
}

}}

#endif
