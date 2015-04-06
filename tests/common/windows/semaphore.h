#ifndef GMAC_TESTS_COMMON_WINDOWS_SEMAPHORE_H_
#define GMAC_TESTS_COMMON_WINDOWS_SEMAPHORE_H_

#include <windows.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int value;
    CONDITION_VARIABLE cond;
    CRITICAL_SECTION mutex;
} gmac_sem_t;


#ifdef __cplusplus
}
#endif

#endif
