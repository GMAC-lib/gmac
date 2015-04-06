#include "Trace.h"
#include "Pcf.h"

#include <config/debug.h>
#include <config/common.h>


#if defined(_MSC_VER)
static inline const char *__getenv(const char *name)
{
	static char buffer[512];
	size_t size = 0;
	if(getenv_s(&size, buffer, 512, name) != 0) return NULL;
	if(strlen(buffer) == 0) return NULL;
	return (const char *)buffer;
}
#else
#define __getenv getenv
#endif

namespace paraver {

Trace *trace = NULL;
int init = 0;
static const char *paraverVar = "PARAVER_OUTPUT";
static const char *defaultOut = "paraver";

CONSTRUCTOR(init);
static void init(void)
{
	TRACE("Paraver Tracing");
	const char *__file = __getenv(paraverVar);
	if(__file == NULL) __file = defaultOut;
	std::string file = std::string(__file) + ".prb";
	trace = new paraver::Trace(file.c_str());
	init = 0;
}

DESTRUCTOR(fini);
static void fini(void)
{
	const char *__file = __getenv(paraverVar);
	if(__file == NULL) __file = defaultOut;
	std::string file = std::string(__file) + ".pcf";
	std::ofstream of(file.c_str(), std::ios::out);
	paraver::pcf(of);
	of.close();

	paraver::Factory<StateName>::destroy();
	paraver::Factory<EventName>::destroy();

	trace->write();
	delete trace;
}


};

// Do not remove!
extern "C" {
void paraver_lib_present()
{
    paraver::trace = NULL;
}
}

#if defined(_WIN32)
#include <windows.h>

// DLL entry function (called on load, unload, ...)
BOOL APIENTRY DllMain(HANDLE /*hModule*/, DWORD dwReason, LPVOID /*lpReserved*/)
{
	switch(dwReason) {
		case DLL_PROCESS_ATTACH:
			paraver::init();
			break;
		case DLL_PROCESS_DETACH:
			paraver::fini();
			break;
		case DLL_THREAD_ATTACH:
			break;
		case DLL_THREAD_DETACH:			
			break;
	};
    return TRUE;
}
#endif
