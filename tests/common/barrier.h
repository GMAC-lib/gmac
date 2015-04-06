#ifndef GMAC_TESTS_COMMON_BARRIER_H_
#define GMAC_TESTS_COMMON_BARRIER_H_

#if defined(POSIX)
#include "posix/barrier.h"
#elif defined(WINDOWS)
#include "windows/barrier.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

void barrier_init(barrier_t *, int);
void barrier_wait(barrier_t *);
void barrier_destroy(barrier_t *);

#ifdef __cplusplus
}
#endif

#endif
