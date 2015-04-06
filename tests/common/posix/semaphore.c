#include "semaphore.h"


void gmac_sem_init(gmac_sem_t *sem, int value)
{
    pthread_cond_init(&sem->cond, NULL);
    pthread_mutex_init(&sem->mutex, NULL);
    sem->value = value;
}

void gmac_sem_post(gmac_sem_t *sem, int v)
{
    int i;
    pthread_mutex_lock(&(sem->mutex));
    sem->value +=v;
    for(i = 0; i < v; i++) {
        pthread_cond_signal(&sem->cond);
    }
    pthread_mutex_unlock(&(sem->mutex));
}

void gmac_sem_wait(gmac_sem_t *sem, int v)
{
    pthread_mutex_lock(&sem->mutex);

    sem->value -= v;
    while(sem->value < 0) {
        pthread_cond_wait(&sem->cond, &sem->mutex);
    }

    pthread_mutex_unlock(&sem->mutex);
}

void gmac_sem_destroy(gmac_sem_t *sem)
{
    pthread_mutex_destroy(&sem->mutex);
    pthread_cond_destroy(&sem->cond);
}
