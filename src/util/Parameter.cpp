#include "Parameter.h"

#include <iostream>
#include <map>

// This CPP directives will generate parameter constructors and declare
// parameters to be used
#undef PARAM
#define PARAM(v, t, d, ...) \
    t v = d; \
    __impl::util::Parameter<t> *__##v = NULL; \
    __impl::util::__Parameter *__init__##v() { \
        __##v = new __impl::util::Parameter<t>(&v, #v, d, ##__VA_ARGS__);\
        return __##v;\
    }
namespace __impl { namespace util { namespace params {
#include "util/Parameter-def.h"

// This CPP directives will create the constructor table for all
// parameters defined by the programmer
#undef PARAM
#define PARAM(v, t, d, ...) \
    { __init__##v, __##v },
ParameterCtor ParamCtorList[] = {
#include "util/Parameter-def.h"
    {NULL, NULL}
};

CONSTRUCTOR(init);
static void init()
{
    for(int i = 0; ParamCtorList[i].ctor != NULL; i++)
        ParamCtorList[i].param = ParamCtorList[i].ctor();

    if(configPrintParams == true) {
        for(int i = 0; ParamCtorList[i].ctor != NULL; i++)
            ParamCtorList[i].param->print();
    }

    // TODO: check this deletion. valgrind shows some errors
    for(int i = 0; ParamCtorList[i].ctor != NULL; i++)
        delete ParamCtorList[i].param;
}


}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
