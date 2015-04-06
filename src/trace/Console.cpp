#if defined(USE_TRACE_CONSOLE)
#include "Console.h"

#include <iostream>

namespace __impl { namespace trace {

void InitApiTracer()
{
	tracer = new __impl::trace::Console();
}
void FiniApiTracer()
{
	if(tracer != NULL) delete tracer;
}

Console::Console() :
	os(std::cerr)
{}

Console::~Console()
{}

void Console::startThread(uint64_t t, THREAD_T tid, const char *name)
{
	os << "@THREAD:START:" << t << ":" << tid << ":" << name << "@" << std::endl;
}

void Console::endThread(uint64_t t, THREAD_T tid)
{
	os << "@THREAD:END:" << t << ":" << tid << "@" << std::endl;
}

void Console::enterFunction(uint64_t t, THREAD_T tid, const char *name)
{
	os << "@FUNCTION:START:" << t << ":" << tid << ":" << name << "@" << std::endl;
}

void Console::exitFunction(uint64_t t, THREAD_T tid, const char *name)
{
	os << "@FUNCTION:END:" << t << ":" << tid << ":" << name << "@" << std::endl;
}

void Console::setThreadState(uint64_t t, THREAD_T tid, State state)
{
	os << "@STATE:" << t << ":" << tid << ":" << state << "@" << std::endl;
}

void Console::dataCommunication(uint64_t t, THREAD_T src, THREAD_T dst, uint64_t delta, size_t size)
{
    os << "@COMM:" << t - delta << ":" << src << ":" << t << ":" << dst << ":" << size << "@" << std::endl;
}

}}

#endif
