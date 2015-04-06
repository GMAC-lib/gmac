#ifndef VARIABLE_H
#define VARIABLE_H

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

typedef std::vector<std::string> VectorString;

class Variable
{
private:
    std::string name_;
    VectorString values_;

public:
    Variable()
    {
    }

    Variable(std::string name) :
        name_(name)
    {
    }

    template <typename T>
    Variable &operator+=(T value)
    {
        std::stringstream stream;
        stream << value;
        values_.push_back(stream.str());
        return *this;
    }

    std::string getName() const
    {
        return name_;
    }

    VectorString::iterator begin()
    {
        VectorString::iterator ret = values_.begin();
        return ret;
    }

    VectorString::iterator end()
    {
        VectorString::iterator ret = values_.end();
        return ret;
    }

    VectorString::iterator find(std::string value)
    {
        VectorString::iterator ret = std::find(values_.begin(), values_.end(), value);
        return ret;
    }
};

#endif /* VARIABLE_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
