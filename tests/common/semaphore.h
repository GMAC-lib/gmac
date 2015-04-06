#ifndef GMAC_TESTS_COMMON_SEMAPHORE_H_
#define GMAC_TESTS_COMMON_SEMAPHORE_H_

#if defined(POSIX)
#include "posix/semaphore.h"
#elif defined(WINDOWS)
#include "windows/semaphore.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

void gmac_sem_init(gmac_sem_t *, int);
void gmac_sem_post(gmac_sem_t *, int );
void gmac_sem_wait(gmac_sem_t *, int );
void gmac_sem_destroy(gmac_sem_t *);


#ifdef __cplusplus
}
#endif

#endif
