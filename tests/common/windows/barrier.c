#include "barrier.h"

void barrier_init(barrier_t *barrier, int value)
{
    InitializeConditionVariable(&barrier->cond);
    InitializeCriticalSection(&barrier->mutex);
    barrier->value = value;
    barrier->counter = 0;
}


void barrier_wait(barrier_t *barrier)
{
    EnterCriticalSection(&barrier->mutex);

    barrier->counter++;
    if(barrier->counter == barrier->value) {
        barrier->counter = 0;
        WakeAllConditionVariable(&barrier->cond);
    }
    else {
        SleepConditionVariableCS(&barrier->cond, &barrier->mutex, INFINITE);
    }

    LeaveCriticalSection(&barrier->mutex);
}

void barrier_destroy(barrier_t *barrier)
{
    DeleteCriticalSection(&barrier->mutex);
}
