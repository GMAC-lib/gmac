#ifndef GMAC_TESTS_COMMON_WINDOWS_BARRIER_H_
#define GMAC_TESTS_COMMON_WINDOWS_BARRIER_H_

#include <windows.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int value;
    int counter;
    CONDITION_VARIABLE cond;
    CRITICAL_SECTION mutex;
} barrier_t;

#ifdef __cplusplus
}
#endif

#endif
