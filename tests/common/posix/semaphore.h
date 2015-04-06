#ifndef GMAC_TESTS_COMMON_POSIX_SEMAPHORE_H_
#define GMAC_TESTS_COMMON_POSIX_SEMAPHORE_H_

#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int value;
    pthread_cond_t cond;
    pthread_mutex_t mutex;
} gmac_sem_t;


#ifdef __cplusplus
}
#endif

#endif
