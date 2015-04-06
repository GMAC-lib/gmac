#ifndef GMAC_TESTS_COMMON_POSIX_BARRIER_H_
#define GMAC_TESTS_COMMON_POSIX_BARRIER_H_

#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int value;
    int counter;
    pthread_cond_t cond;
    pthread_mutex_t mutex;
} barrier_t;

#ifdef __cplusplus
}
#endif

#endif
