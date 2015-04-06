#include "../utils.h"

#include <sys/time.h>


void getTime(gmactime_t *out)
{
	if(out == NULL) return;
    struct timeval tv;
    if(gettimeofday(&tv, NULL) < 0) return;
    out->usec = tv.tv_usec;
    out->sec = tv.tv_sec;
}

thread_t thread_create(thread_routine rtn, void *arg)
{
    pthread_t tid;
    int ret  = pthread_create(&tid, NULL, rtn, arg);
    if(ret != 0) return (thread_t)0;
    return tid;
}

void thread_wait(thread_t id)
{
    pthread_join(id, NULL);
}
