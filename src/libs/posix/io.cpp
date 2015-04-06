#if !defined(_MSC_VER)
#include <unistd.h>
#include <stdint.h>
#include <dlfcn.h>
#endif

#include <cstdio>
#include <errno.h>

#include "core/IOBuffer.h"
#include "core/hpe/Mode.h"
#include "core/hpe/Process.h"
#include "libs/common.h"
#include "memory/Manager.h"
#include "util/loader.h"

#include "posix.h"

using namespace __impl::core;
using namespace __impl::memory;
using __impl::util::params::ParamBlockSize;

SYM(ssize_t, __libc_read, int, void *, size_t);
SYM(ssize_t, __libc_write, int, const void *, size_t);

/* System call wrappers */

#ifdef __cplusplus
extern "C"
#endif
ssize_t SYMBOL(read)(int fd, void *buf, size_t count)
{
	if(__libc_read == NULL) posixIoInit();
	if(inGmac() == 1 || count == 0) return __libc_read(fd, buf, count);

    enterGmac();
    Mode *dstMode = getProcess().owner(hostptr_t(buf));

    if(dstMode == NULL) {
        exitGmac();
        return __libc_read(fd, buf, count);
    }

	gmac::trace::SetThreadState(gmac::trace::IO);
    gmacError_t err;
    ssize_t ret = 0;
    size_t bufferSize = ParamBlockSize > count ? ParamBlockSize : count;
    Mode &mode = getMode(*dstMode);
    IOBuffer *buffer1 = &mode.createIOBuffer(bufferSize, GMAC_PROT_READ);
    IOBuffer *buffer2 = NULL;
    if (count > buffer1->size()) {
        buffer2 = &mode.createIOBuffer(bufferSize, GMAC_PROT_READ);
    }

    Manager &manager = getManager();
    IOBuffer *active  = buffer1;
    IOBuffer *passive = buffer2;

    size_t left = count;
    size_t  off  = 0;
    while (left != 0) {
        err = active->wait();
        ASSERTION(err == gmacSuccess);
        size_t bytes= left < active->size()? left: active->size();
        ret += __libc_read(fd, active->addr(), bytes);
        ret = manager.fromIOBuffer(mode, hostptr_t(buf) + off, *active, 0, bytes);
        ASSERTION(ret == gmacSuccess);

        left -= bytes;
        off  += bytes;
        IOBuffer *tmp = active;
        active = passive;
        passive = tmp;
    }
    err = passive->wait();
    ASSERTION(err == gmacSuccess);
    mode.destroyIOBuffer(*buffer1);
    if (buffer2 != NULL) {
        mode.destroyIOBuffer(*buffer2);
    }
	gmac::trace::SetThreadState(gmac::trace::Running);
	exitGmac();

    return ret;
}

#ifdef __cplusplus
extern "C"
#endif
ssize_t SYMBOL(write)(int fd, const void *buf, size_t count)
{
	if(__libc_read == NULL) posixIoInit();
	if(inGmac() == 1 || count == 0) return __libc_write(fd, buf, count);

	enterGmac();
    Mode *srcMode = getProcess().owner(hostptr_t(buf));

    if(srcMode == NULL) {
        exitGmac();
        return __libc_write(fd, buf, count);
    }

	gmac::trace::SetThreadState(gmac::trace::IO);
    gmacError_t err;
    ssize_t ret = 0;

    size_t off  = 0;
    size_t bufferSize = ParamBlockSize > count ? ParamBlockSize : count;
    Mode &mode = getMode(*srcMode);
    IOBuffer *buffer1 = &mode.createIOBuffer(bufferSize, GMAC_PROT_READ);
    IOBuffer *buffer2 = NULL;
    if (count > buffer1->size()) {
        buffer2 = &mode.createIOBuffer(bufferSize, GMAC_PROT_READ);
    }

    Manager &manager = getManager();
    IOBuffer *active  = buffer1;
    IOBuffer *passive = buffer2;

    size_t left = count;

    size_t bytesActive = left < active->size() ? left : active->size();
    err = manager.toIOBuffer(mode, *active, 0, hostptr_t(buf) + off, bytesActive);
    ASSERTION(err == gmacSuccess);
    size_t bytesPassive = 0;

    do {
        left -= bytesActive;
        off  += bytesActive;

        if (left > 0) {
            bytesPassive = left < passive->size()? left : passive->size();
            err = manager.toIOBuffer(mode, *passive, 0, hostptr_t(buf) + off, bytesPassive);
            ASSERTION(err == gmacSuccess);
        }

        err = active->wait();
        ASSERTION(err == gmacSuccess);

        ret += __libc_write(fd, active->addr(), bytesActive);

        size_t bytesTmp = bytesActive;
        bytesActive = bytesPassive;
        bytesPassive = bytesTmp;
        
        IOBuffer *tmp = active;
        active = passive;
        passive = tmp;
    } while (left != 0);
    ASSERTION(err == gmacSuccess);
    mode.destroyIOBuffer(*buffer1);
    if (buffer2 != NULL) {
        mode.destroyIOBuffer(*buffer2);
    }
	gmac::trace::SetThreadState(gmac::trace::Running);
	exitGmac();

    return ret;
}

void posixIoInit(void)
{
	TRACE(GLOBAL, "Overloading I/O POSIX functions");
	LOAD_SYM(__libc_read, read);
	LOAD_SYM(__libc_write, write);
}
