#ifndef GMAC_UTIL_POSIX_FILELOCK_IMPL_H_
#define GMAC_UTIL_POSIX_FILELOCK_IMPL_H_

#include <sys/file.h>
#include <errno.h>

#include "util/Logger.h"

namespace __impl { namespace util {

inline void
FileLock::lock()
{
    int ret;
    enter();
    ret = flock(_fd, LOCK_EX);
    ASSERTION(ret == 0, "Error locking file: %s", strerr(errno));
    locked();
}

inline void
FileLock::unlock()
{
    int ret;
    exit();
    ret = flock(_fd, LOCK_UN);
    ASSERTION(ret == 0, "Error unlocking file: %s", strerr(errno));
}


inline FILE *
FileLock::file()
{
    return _file;
}

}}

#endif
