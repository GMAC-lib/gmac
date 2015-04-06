#include "utils.h"

double getTimeStamp(gmactime_t time)
{
	double t;
	t = 1e6 * time.sec + (time.usec);
	return t;
}

void printTime(gmactime_t *start, gmactime_t *end, const char *pre, const char *post)
{
	double s, e;
	s = 1e6 * start->sec + (start->usec);
	e = 1e6 * end->sec + (end->usec);
	printf("%s%f%s", pre, (e - s) / 1e6, post);
}

void printAvgTime(gmactime_t *start, gmactime_t *end, const char *pre, const char *post, unsigned rounds)
{
	double s, e;
	s = 1e6 * start->sec + (start->usec);
	e = 1e6 * end->sec + (end->usec);
	printf("%s%f%s", pre, (e - s) / 1e6 / rounds, post);
}

void randInit(float *a, size_t size)
{
	for(unsigned i = 0; i < size; i++) {
		a[i] = 1.0f * rand();
	}
}

void randInitMax(float *a, float maxVal, size_t size)
{
	for(unsigned i = 0; i < size; i++) {
		a[i] = 1.f * (rand() % int(maxVal));
	}
}

void valueInit(float *a, float v, size_t size)
{
	for(unsigned i = 0; i < size; i++) {
		a[i] = v;
	}
}

#if defined(__GNUC__)
	void utils_init(void) __attribute__ ((constructor));
#	define CTOR
#elif defined(_MSC_VER)
#	pragma section(".CRT$XCU",read)
	static void __cdecl utils_init(void); 
	__declspec(allocate(".CRT$XCU")) void (__cdecl*utils_init_)(void) = utils_init;
#	define CTOR __cdecl
#endif

#include <ctime>
void CTOR utils_init()
{
    srand((unsigned int)time(NULL));
}
