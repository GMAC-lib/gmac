#include "../utils.h"

#include <windows.h>

thread_t thread_create(thread_routine rtn, void *arg)
{
	HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)rtn, arg, 0, NULL);
	thread_t ret = GetThreadId(handle);
	CloseHandle(handle);
	return ret;
}

void thread_wait(thread_t id)
{
	HANDLE handle = OpenThread(SYNCHRONIZE, FALSE, id);
	if (handle == NULL) return;
	WaitForSingleObject(handle, INFINITE);
	CloseHandle(handle);
}
