#ifdef USE_DBC

#include <iostream>
#include <cassert>

#include "Contract.h"

namespace __dbc {

void Contract::Preamble(const char *file, const int line)
{
    std::cerr << "[Breach of Contract @ " << file << ":" << line << " ";
}


void Contract::Ensures(const char *file, const int line,
        const char *clause, bool b)
{
    if(b == true) return;
    Preamble(file, line);
    std::cerr << "Ensured " << clause << " not met" << std::endl;
    assert(0);
}

void Contract::Requires(const char *file, const int line,
        const char *clause, bool b)
{
    if(b == true) return;
    Preamble(file, line);
    std::cerr << "Required " << clause << " not met" << std::endl;
    assert(0);
}

void Contract::Expects(const char *file, const int line,
        const char *clause, bool b)
{
    if(b == true) return;
    Preamble(file, line);
    std::cerr << "Expected " << clause << " not met" << std::endl;
}


void Contract::Assert(const char *file, const int line,
        const char *clause, bool b)
{
    if(b == true) return;
    Preamble(file, line);
    std::cerr << "Assert " << clause << " not met" << std::endl;
    assert(0);
}

}

#endif
