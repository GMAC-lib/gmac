#include <cstdio>
#include <errno.h>

#include "core/IOBuffer.h"
#include "core/Process.h"
#include "core/Mode.h"

#include "libs/common.h"

#include "memory/Manager.h"

#include "trace/Tracer.h"

#include "util/loader.h"
#include "util/Logger.h"

#include "stdc.h"

using namespace __impl::core;
using namespace __impl::memory;
using __impl::util::params::ParamBlockSize;

SYM(size_t, __libc_fread, void *, size_t, size_t, FILE *);
SYM(size_t, __libc_fwrite, const void *, size_t, size_t, FILE *);

#ifdef __cplusplus
extern "C"
#endif
size_t SYMBOL(fread)(void *buf, size_t size, size_t nmemb, FILE *stream)
{
	if(__libc_fread == NULL) stdcIoInit();
	if((inGmac() == 1) ||
       (size * nmemb == 0)) return __libc_fread(buf, size, nmemb, stream);

    enterGmac();
    Mode *dstMode = getProcess().owner(hostptr_t(buf), size);

    if(dstMode == NULL) {
        exitGmac();
        return  __libc_fread(buf, size, nmemb, stream);
    }

	gmac::trace::SetThreadState(gmac::trace::IO);
    gmacError_t err;
    size_t n = size * nmemb;
    size_t ret = 0;

    size_t off = 0;
    size_t bufferSize = ParamBlockSize > size ? ParamBlockSize : size;
    Mode &mode = getMode(*dstMode);
    IOBuffer *buffer1 = &mode.createIOBuffer(bufferSize, GMAC_PROT_READ);
    IOBuffer *buffer2 = NULL;
    if (n > buffer1->size()) {
        buffer2 = &mode.createIOBuffer(bufferSize, GMAC_PROT_READ);
    }

    Manager &manager = getManager();
    IOBuffer *active  = buffer1;
    IOBuffer *passive = buffer2;

    size_t left = n;
    while (left != 0) {
        err = active->wait();
        ASSERTION(err == gmacSuccess);
        size_t bytes = left < active->size()? left: active->size();
        size_t elems = __libc_fread(active->addr(), size, bytes/size, stream);
        if(elems == 0) break;
		ret += elems;
        err = manager.fromIOBuffer(mode, (uint8_t *)buf + off, *active, 0, size * elems);
        ASSERTION(err == gmacSuccess);

        left -= size * elems;
        off  += size * elems;
        TRACE(GLOBAL, FMT_SIZE" of "FMT_SIZE" bytes read", elems * size, nmemb * size);
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
size_t SYMBOL(fwrite)(const void *buf, size_t size, size_t nmemb, FILE *stream)
{
    if(__libc_fwrite == NULL) stdcIoInit();
	if((inGmac() == 1) ||
       (size * nmemb == 0)) return __libc_fwrite(buf, size, nmemb, stream);

	enterGmac();
    Mode *srcMode = getProcess().owner(hostptr_t(buf), size);

    if(srcMode == NULL) {
        exitGmac();
        return __libc_fwrite(buf, size, nmemb, stream);
    }

	gmac::trace::SetThreadState(gmac::trace::IO);
    gmacError_t err;
    size_t n = size * nmemb;
    size_t ret = 0;

    size_t off = 0;
    size_t bufferSize = ParamBlockSize > size ? ParamBlockSize : size;
    Mode &mode = getMode(*srcMode);
    IOBuffer *buffer1 = &mode.createIOBuffer(bufferSize, GMAC_PROT_READ);
    IOBuffer *buffer2 = NULL;
    if (n > buffer1->size()) {
        buffer2 = &mode.createIOBuffer(bufferSize, GMAC_PROT_READ);
    }

    Manager &manager = getManager();
    IOBuffer *active  = buffer1;
    IOBuffer *passive = buffer2;

    size_t left = n;

    size_t bytesActive = left < active->size()? left : active->size();
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

        size_t elems = __libc_fwrite(active->addr(), size, bytesActive/size, stream);
        if(elems == 0) break;
        TRACE(GLOBAL, FMT_SIZE" of "FMT_SIZE" bytes written", elems * size, nmemb * size);
        ret += elems;

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

void stdcIoInit(void)
{
	LOAD_SYM(__libc_fread, fread);
	LOAD_SYM(__libc_fwrite, fwrite);
}
