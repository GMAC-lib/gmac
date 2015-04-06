#if defined(USE_TRACE)
#include "trace/Tracer.h"
#include <windows.h>

#include "trace/Tracer.h"

#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64

namespace __impl { namespace trace {

static LARGE_INTEGER TicksPerMicroSecond_ = { 0 };

#if defined(USE_TRACE)
uint64_t Tracer::timeMark() const
{
	LARGE_INTEGER tick;
	if(TicksPerMicroSecond_.QuadPart == 0) {
		QueryPerformanceFrequency(&TicksPerMicroSecond_);
		TicksPerMicroSecond_.QuadPart /= 1000000;
	}
	QueryPerformanceCounter(&tick);
	uint64_t ret = tick.QuadPart / TicksPerMicroSecond_.QuadPart;

	return ret - base_;
}
#endif

}}
#endif