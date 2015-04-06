#include "barrier.h"

void barrier_init(barrier_t *barrier, int value)
{
    pthread_cond_init(&barrier->cond, NULL);
    pthread_mutex_init(&barrier->mutex, NULL);
    barrier->value = value;
    barrier->counter = 0;
}


void barrier_wait(barrier_t *barrier)
{
    pthread_mutex_lock(&barrier->mutex);

    barrier->counter++;
    if(barrier->counter == barrier->value) {
        barrier->counter = 0;
        pthread_cond_broadcast(&barrier->cond);
    }
    else {
        pthread_cond_wait(&barrier->cond, &barrier->mutex);
    }

    pthread_mutex_unlock(&barrier->mutex);
}

void barrier_destroy(barrier_t *barrier)
{
    pthread_mutex_destroy(&barrier->mutex);
    pthread_cond_destroy(&barrier->cond);
}
