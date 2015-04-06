#include "semaphore.h"
#include <windows.h>

void gmac_sem_init(gmac_sem_t *sem, int value)
{
    InitializeConditionVariable(&sem->cond);
    InitializeCriticalSection(&sem->mutex);
    sem->value = value;
}

void gmac_sem_post(gmac_sem_t *sem, int v)
{
    int i;
    EnterCriticalSection(&sem->mutex);

    sem->value += v;
    for(i = 0; i < v; i++)
        WakeConditionVariable(&sem->cond);

    LeaveCriticalSection(&sem->mutex);
}

void gmac_sem_wait(gmac_sem_t *sem, int v)
{
    EnterCriticalSection(&sem->mutex);

    sem->value -= v;
    while(sem->value < 0) {
        SleepConditionVariableCS(&sem->cond, &sem->mutex, INFINITE);
    }

    LeaveCriticalSection(&sem->mutex);
}

void gmac_sem_destroy(gmac_sem_t *sem)
{
    DeleteCriticalSection(&sem->mutex);
}
