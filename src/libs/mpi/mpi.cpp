#ifdef USE_MPI

#include "libs/common.h"
#include "memory/Manager.h"
#include "core/IOBuffer.h"
#include "core/Process.h"
#include "core/Mode.h"

#include "util/loader.h"

#include "mpi_local.h"

using __impl::core::IOBuffer;
using __impl::core::Mode;
using __impl::core::Process;

using __impl::memory::getManager;
using __impl::core::getMode;
using __impl::core::getProcess;

using __impl::memory::Manager;

SYM(int, __MPI_Sendrecv, void *, int, MPI_Datatype, int, int, void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status *);

SYM(int, __MPI_Send    , void *, int, MPI_Datatype, int, int, MPI_Comm );
SYM(int, __MPI_Ssend   , void *, int, MPI_Datatype, int, int, MPI_Comm );
SYM(int, __MPI_Rsend   , void *, int, MPI_Datatype, int, int, MPI_Comm );
SYM(int, __MPI_Bsend   , void *, int, MPI_Datatype, int, int, MPI_Comm );

SYM(int, __MPI_Recv    , void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status *);

void mpiInit(void)
{
	TRACE(GLOBAL, "Overloading MPI_Sendrecv");
	LOAD_SYM(__MPI_Sendrecv, MPI_Sendrecv);

	LOAD_SYM(__MPI_Send,  MPI_Send);
	LOAD_SYM(__MPI_Ssend, MPI_Ssend);
	LOAD_SYM(__MPI_Rsend, MPI_Rsend);
	LOAD_SYM(__MPI_Bsend, MPI_Bsend);

	LOAD_SYM(__MPI_Recv,  MPI_Recv);
}

extern "C"
int MPI_Sendrecv( void *sendbuf, int sendcount, MPI_Datatype sendtype, 
        int dest, int sendtag, 
        void *recvbuf, int recvcount, MPI_Datatype recvtype, 
        int source, int recvtag, MPI_Comm comm, MPI_Status *status )
{
	if(inGmac() == 1) return __MPI_Sendrecv(sendbuf, sendcount, sendtype, dest,   sendtag,
                                                  recvbuf, recvcount, recvtype, source, recvtag, comm, status);
    if(__MPI_Sendrecv == NULL) mpiInit();

    Process &proc = getProcess();
	// Check if GMAC owns any of the buffers to be transferred
	Mode *srcMode = proc.owner(hostptr_t(sendbuf));
	Mode *dstMode = proc.owner(hostptr_t(recvbuf));

	if (srcMode == NULL && dstMode == NULL) {
        return __MPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status);
    }

    enterGmac();

    Mode &mode = getMode(*dstMode);
    Manager &manager = getManager();

    gmacError_t err;
    int ret, ret2;

    void * tmpSend = NULL;
    void * tmpRecv = NULL;

    bool allocSend = false;
    bool allocRecv = false;

    int typebytes;
    size_t sendbytes = 0;
    size_t recvbytes = 0;
    size_t bufferUsed = 0;

    ret  = MPI_Type_size(sendtype, &typebytes);
    sendbytes = sendcount * typebytes;
    ret2 = MPI_Type_size(recvtype, &typebytes);
    recvbytes = recvcount * typebytes;
    IOBuffer &buffer = mode.createIOBuffer(recvbytes + sendbytes, GMAC_PROT_READWRITE);

    if (ret  != MPI_SUCCESS) goto exit;
    if (ret2 != MPI_SUCCESS) goto exit;

    ASSERTION(buffer.size() >= size_t(recvbytes + sendbytes));

    if(dest != MPI_PROC_NULL && srcMode != NULL) {
        // Fast path
        if (buffer.size() >= sendbytes) {
            tmpSend     = buffer.addr();
            bufferUsed += sendbytes;

            err = manager.toIOBuffer(mode, buffer, 0, hostptr_t(sendbuf), sendbytes);
            ASSERTION(err == gmacSuccess);
            err = buffer.wait();
            ASSERTION(err == gmacSuccess);
        } // Slow path
        else {
            // Alloc buffer
            tmpSend = malloc(sendbytes);
            ASSERTION(tmpSend != NULL);
            allocSend = true;

            err = manager.memcpy(mode, hostptr_t(tmpSend), hostptr_t(sendbuf), sendbytes);
            ASSERTION(err == gmacSuccess);
        }
	} else {
        tmpSend = sendbuf;
    }

    if(source != MPI_PROC_NULL && dstMode != NULL) {
        if (buffer.size() - bufferUsed >= recvbytes) {
            tmpRecv = buffer.addr() + bufferUsed;
        } else {
            // Alloc buffer
            tmpRecv = malloc(recvbytes);
            ASSERTION(tmpRecv != NULL);

            allocRecv = true;
        }
    } else {
        tmpRecv = recvbuf;
    }

    ret = __MPI_Sendrecv(tmpSend, sendcount, sendtype, dest,   sendtag,
                         tmpRecv, recvcount, recvtype, source, recvtag, comm, status);

    if (allocSend) {
        free(tmpSend);
    }

    if(source != MPI_PROC_NULL && dstMode != NULL) {
        if (!allocRecv) {
            err = manager.fromIOBuffer(mode, hostptr_t(recvbuf), buffer, bufferUsed, recvbytes);
            ASSERTION(err == gmacSuccess);
            err = buffer.wait();
            ASSERTION(err == gmacSuccess);
        } else {
            err = manager.memcpy(mode, hostptr_t(recvbuf), hostptr_t(tmpRecv), recvbytes);
            ASSERTION(err == gmacSuccess);

            // Free temporal buffer
            free(tmpRecv);
        }
    }

exit:
    mode.destroyIOBuffer(buffer);

	exitGmac();

    return ret;
}

extern "C"
int __gmac_MPI_Send( void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
    int (*func)(void *, int, MPI_Datatype, int, int, MPI_Comm))
{
	if(inGmac() == 1) return func(buf, count, datatype, dest, tag, comm);
    if(__MPI_Send == NULL) mpiInit();

	// Check if GMAC owns any of the buffers to be transferred
	Mode *srcMode = getProcess().owner(hostptr_t(buf));

	if (srcMode == NULL) {
        return func(buf, count, datatype, dest, tag, comm);
    }

    enterGmac();

    Mode &mode = getMode(*srcMode);
    Manager &manager = getManager();

    gmacError_t err;
    int ret;

    void * tmpSend = NULL;

    bool allocSend = false;

    int typebytes;
    size_t sendbytes = 0;

    ret = MPI_Type_size(datatype, &typebytes);
    sendbytes = count * typebytes;
    IOBuffer &buffer = mode.createIOBuffer(sendbytes, GMAC_PROT_WRITE);
    if (ret != MPI_SUCCESS) goto exit;

    if (dest != MPI_PROC_NULL) {
        if (buffer.size() >= sendbytes) {
            tmpSend     = buffer.addr();

            err = manager.toIOBuffer(mode, buffer, 0, hostptr_t(buf), sendbytes);
            ASSERTION(err == gmacSuccess);
            err = buffer.wait();
            ASSERTION(err == gmacSuccess);
        } else {
            // Alloc buffer
            tmpSend = malloc(sendbytes);
            ASSERTION(tmpSend != NULL);
            allocSend = true;

            err = manager.memcpy(mode, hostptr_t(tmpSend), hostptr_t(buf), sendbytes);
            ASSERTION(err == gmacSuccess);
        }
	} else {
        tmpSend = buf;
    }

    ret = func(tmpSend, count, datatype, dest, tag, comm);

    if (allocSend) {
        // Free temporal buffer
        free(tmpSend);
    }

exit:
    mode.destroyIOBuffer(buffer);

	exitGmac();

    return ret;
}


extern "C"
int MPI_Send ( void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm )
{
    return __gmac_MPI_Send(buf, count, datatype, dest, tag, comm, __MPI_Send);
}

extern "C"
int MPI_Ssend( void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm )
{
    return __gmac_MPI_Send(buf, count, datatype, dest, tag, comm, __MPI_Ssend);
}

extern "C"
int MPI_Rsend( void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm )
{
    return __gmac_MPI_Send(buf, count, datatype, dest, tag, comm, __MPI_Rsend);
}

extern "C"
int MPI_Bsend( void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm )
{
    return __gmac_MPI_Send(buf, count, datatype, dest, tag, comm, __MPI_Bsend);
}

extern "C"
int MPI_Recv( void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status )
{
	if(inGmac() == 1) return __MPI_Recv(buf, count, datatype, source, tag, comm, status);
    if(__MPI_Recv == NULL) mpiInit();

	// Locate memory regions (if any)
	Mode *dstMode = getProcess().owner(hostptr_t(buf));

	if (dstMode == NULL) {
        return __MPI_Recv(buf, count, datatype, source, tag, comm, status);
    }

    enterGmac();

    Mode &mode = getMode(*dstMode);
    Manager &manager = getManager();

    gmacError_t err;
    int ret;

    void * tmpRecv = NULL;

    bool allocRecv = false;

    int typebytes;
    size_t recvbytes = 0;
    
    ret = MPI_Type_size(datatype, &typebytes);
    recvbytes = count * typebytes;
    IOBuffer &buffer = mode.createIOBuffer(recvbytes, GMAC_PROT_READ);
    if (ret != MPI_SUCCESS) goto exit;

    if(source != MPI_PROC_NULL) {
        if (buffer.size() >= recvbytes) {
            tmpRecv = buffer.addr();
        } else {
            // Alloc buffer
            tmpRecv = malloc(recvbytes);
            ASSERTION(tmpRecv != NULL);
            allocRecv = true;
        }
    } else {
        tmpRecv = buf;
    }

    ret = __MPI_Recv(tmpRecv, count, datatype, source, tag, comm, status);

    if(source != MPI_PROC_NULL) {
        if (!allocRecv) {
            err = manager.fromIOBuffer(mode, hostptr_t(buf), buffer, 0, recvbytes);
            ASSERTION(err == gmacSuccess);
            err = buffer.wait();
            ASSERTION(err == gmacSuccess);
        } else {
            err = manager.memcpy(mode, hostptr_t(buf), hostptr_t(tmpRecv), recvbytes);
            ASSERTION(err == gmacSuccess);

            // Free temporal buffer
            free(tmpRecv);
        }
    }

    if (allocRecv) {
        // Free temporal buffer
        free(tmpRecv);
    }

exit:
    mode.destroyIOBuffer(buffer);

	exitGmac();

    return ret;
}

#endif
