#include "FileLock.h"

#if 0
namespace __impl { namespace util {

FileLock::FileLock(const char * fname, const char *_name) :
    __impl::util::__Lock(_name)
{
    _file = fopen(fname, "rw");
    ASSERTION(_file != NULL, "Error opening file '%s' for lock", fname);
    _fd = fileno(_file);
}

FileLock::~FileLock()
{
    fclose(_file);
}

}}

#endif
