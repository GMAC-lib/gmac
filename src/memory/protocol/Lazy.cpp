#include "Lazy.h"

#include "core/IOBuffer.h"

#include "config/config.h"

#include "memory/Memory.h"
#include "memory/StateBlock.h"

#include "trace/Tracer.h"

#ifdef DEBUG
#include <ostream>
#endif


#if defined(__GNUC__)
#define MIN std::min
#elif defined(_MSC_VER)
#define MIN min
#endif

namespace __impl { namespace memory { namespace protocol {


LazyBase::LazyBase(bool eager) :
    gmac::util::Lock("LazyBase"),
    eager_(eager),
    limit_(1)
{
}

LazyBase::~LazyBase()
{
}

lazy::State LazyBase::state(GmacProtection prot) const
{
    switch(prot) {
    case GMAC_PROT_NONE:
        return lazy::Invalid;
    case GMAC_PROT_READ:
        return lazy::ReadOnly;
    case GMAC_PROT_WRITE:
    case GMAC_PROT_READWRITE:
            return lazy::Dirty;
    }
    return lazy::Dirty;
}


void LazyBase::deleteObject(Object &obj)
{
    obj.decRef();
}

bool LazyBase::needUpdate(const Block &b) const
{
    const lazy::Block &block = dynamic_cast<const lazy::Block &>(b);
    switch(block.getState()) {
    case lazy::Dirty:
    case lazy::HostOnly:
        return false;
    case lazy::ReadOnly:
    case lazy::Invalid:
        return true;
    }
    return false;
}

gmacError_t LazyBase::signalRead(Block &b, hostptr_t addr)
{
    trace::EnterCurrentFunction();
    lazy::Block &block = dynamic_cast<lazy::Block &>(b);
    gmacError_t ret = gmacSuccess;

    block.read(addr);
    if(block.getState() == lazy::HostOnly) {
        WARNING("Signal on HostOnly block - Changing protection and continuing");
        if(block.unprotect() < 0)
            FATAL("Unable to set memory permissions");

        goto exit_func;
    }

    if (block.getState() == lazy::Invalid) {
        ret = block.syncToHost();
        if(ret != gmacSuccess) goto exit_func;
        block.setState(lazy::ReadOnly);
    }

    if(block.protect(GMAC_PROT_READ) < 0)
        FATAL("Unable to set memory permissions");

exit_func:
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t LazyBase::signalWrite(Block &b, hostptr_t addr)
{
    trace::EnterCurrentFunction();
    lazy::Block &block = dynamic_cast<lazy::Block &>(b);
    gmacError_t ret = gmacSuccess;

    block.write(addr);
    switch (block.getState()) {
    case lazy::Dirty:
        block.unprotect();
        goto exit_func; // Somebody already fixed it
    case lazy::Invalid:
        ret = block.syncToHost();
        if(ret != gmacSuccess) goto exit_func;
        break;
    case lazy::HostOnly:
        WARNING("Signal on HostOnly block - Changing protection and continuing");
    case lazy::ReadOnly:
        break;
    }
    block.setState(lazy::Dirty, addr);
    block.unprotect();
    addDirty(block);
    TRACE(LOCAL,"Setting block %p to dirty state", block.addr());
    //ret = addDirty(block);
exit_func:
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t LazyBase::acquire(Block &b, GmacProtection &prot)
{
    gmacError_t ret = gmacSuccess;
    lazy::Block &block = dynamic_cast<lazy::Block &>(b);
    switch(block.getState()) {
    case lazy::Invalid:
    case lazy::ReadOnly:
        if (prot == GMAC_PROT_READWRITE ||
            prot == GMAC_PROT_WRITE) {
            if(block.protect(GMAC_PROT_NONE) < 0)
                FATAL("Unable to set memory permissions");
#ifndef USE_VM
            block.setState(lazy::Invalid);
            //block.acquired();
#endif
        }

        break;
    case lazy::Dirty:
        WARNING("Block modified before gmacSynchronize: %p", block.addr());
        break;
    case lazy::HostOnly:
        break;
    }
    return ret;
}

#ifdef USE_VM
gmacError_t LazyBase::acquireWithBitmap(Block &b)
{
    /// \todo Change this to the new BlockState
    gmacError_t ret = gmacSuccess;
    lazy::Block &block = dynamic_cast<lazy::Block &>(b);
    switch(block.getState()) {
    case lazy::Invalid:
    case lazy::ReadOnly:
        if (block.is(lazy::Invalid)) {
            if(block.protect(GMAC_PROT_NONE) < 0)
                FATAL("Unable to set memory permissions");
            block.setState(lazy::Invalid);
        } else {
            if(block.protect(GMAC_PROT_READ) < 0)
                FATAL("Unable to set memory permissions");
            block.setState(lazy::ReadOnly);
        }
        break;
    case lazy::Dirty:
        FATAL("Block in incongruent state in acquire: %p", block.addr());
        break;
    case lazy::HostOnly:
        break;
    }
    return ret;
}
#endif

gmacError_t LazyBase::mapToAccelerator(Block &b)
{
    lazy::Block &block = dynamic_cast<lazy::Block &>(b);
    ASSERTION(block.getState() == lazy::HostOnly);
    TRACE(LOCAL,"Mapping block to accelerator %p", block.addr());
    block.setState(lazy::Dirty);
    addDirty(block);
    return gmacSuccess;
}

gmacError_t LazyBase::unmapFromAccelerator(Block &b)
{
    lazy::Block &block = dynamic_cast<lazy::Block &>(b);
    TRACE(LOCAL,"Unmapping block from accelerator %p", block.addr());
    gmacError_t ret = gmacSuccess;
    switch(block.getState()) {
    case lazy::HostOnly:
    case lazy::Dirty:
    case lazy::ReadOnly:
        break;
    case lazy::Invalid:
        ret = block.syncToHost();
        if(ret != gmacSuccess) break;
    }
    if(block.unprotect() < 0)
        FATAL("Unable to set memory permissions");
    block.setState(lazy::HostOnly);
    dbl_.remove(block);
    return ret;
}

void
LazyBase::addDirty(lazy::Block &block)
{
    lock();
    dbl_.push(block);
    if (eager_ == false) {
        unlock();
        return;
    } else {
        if (block.getCacheWriteFaults() >= __impl::util::params::ParamRollThreshold) {
            block.resetCacheWriteFaults();
            TRACE(LOCAL, "Increasing dirty block cache limit -> %u", limit_ + 1);
            limit_++;
        }
    }
    while (dbl_.size() > limit_) {
        Block &b = dbl_.front();
        b.coherenceOp(&Protocol::release);
    }
    unlock();
    return;
}

gmacError_t LazyBase::releaseAll()
{
    // We need to make sure that this operations is done before we
    // let other modes to proceed
    lock();

    // Shrink cache size if we have not filled it
    if (eager_ == true && dbl_.size() < limit_ && limit_ > 1) {
        limit_ /= 2;
    }

    // If the list of objects to be released is empty, assume a complete flush
    TRACE(LOCAL, "Releasing all blocks");

    while(dbl_.empty() == false) {
        Block &b = dbl_.front();
        gmacError_t ret = b.coherenceOp(&Protocol::release);
        ASSERTION(ret == gmacSuccess);
    }

    unlock();
    return gmacSuccess;
}

gmacError_t LazyBase::flushDirty()
{
    return releaseAll();
}


gmacError_t LazyBase::releasedAll()
{
    lock();

    // Shrink cache size if we have not filled it
    if (eager_ == true && dbl_.size() < limit_ && limit_ > 1) {
        TRACE(LOCAL, "Shrinking dirty block cache limit %u -> %u", limit_, limit_ / 2);
        limit_ /= 2;
    }

    unlock();

    return gmacSuccess;
}

gmacError_t LazyBase::release(Block &b)
{
    lazy::Block &block = dynamic_cast<lazy::Block &>(b);
    TRACE(LOCAL,"Releasing block %p", block.addr());
    gmacError_t ret = gmacSuccess;
    switch(block.getState()) {
    case lazy::Dirty:
        if(block.protect(GMAC_PROT_READ) < 0)
            FATAL("Unable to set memory permissions");
        ret = block.syncToAccelerator();
        if(ret != gmacSuccess) break;
        block.setState(lazy::ReadOnly);
        block.released();
        dbl_.remove(block);
        break;
    case lazy::Invalid:
    case lazy::ReadOnly:
    case lazy::HostOnly:
        break;
    }
    return ret;
}

gmacError_t LazyBase::deleteBlock(Block &block)
{
    dbl_.remove(dynamic_cast<lazy::Block &>(block));
    return gmacSuccess;
}

gmacError_t LazyBase::toHost(Block &b)
{
    TRACE(LOCAL,"Sending block to host: %p", b.addr());
    gmacError_t ret = gmacSuccess;
    lazy::Block &block = dynamic_cast<lazy::Block &>(b);
    switch(block.getState()) {
    case lazy::Invalid:
        ret = block.syncToHost();
        TRACE(LOCAL,"Invalid block");
        if(block.protect(GMAC_PROT_READ) < 0)
            FATAL("Unable to set memory permissions");
        if(ret != gmacSuccess) break;
        block.setState(lazy::ReadOnly);
        break;
    case lazy::Dirty:
        TRACE(LOCAL,"Dirty block");
        break;
    case lazy::ReadOnly:
        TRACE(LOCAL,"ReadOnly block");
        break;
    case lazy::HostOnly:
        TRACE(LOCAL,"HostOnly block");
        break;
    }
    return ret;
}

#if 0
gmacError_t LazyBase::toAccelerator(Block &b)
{
    TRACE(LOCAL,"Sending block to accelerator: %p", b.addr());
    gmacError_t ret = gmacSuccess;
    lazy::Block &block = dynamic_cast<lazy::Block &>(b);
    switch(block.getState()) {
    case lazy::Dirty:
        TRACE(LOCAL,"Dirty block");
        if(block.protect(GMAC_PROT_READ) < 0)
            FATAL("Unable to set memory permissions");
        ret = block.syncToAccelerator();
        if(ret != gmacSuccess) break;
        block.setState(lazy::ReadOnly);
        break;
    case lazy::Invalid:
        TRACE(LOCAL,"Invalid block");
        break;
    case lazy::ReadOnly:
        TRACE(LOCAL,"ReadOnly block");
        break;
    case lazy::HostOnly:
        TRACE(LOCAL,"HostOnly block");
        break;
    }
    return ret;
}
#endif

gmacError_t LazyBase::copyToBuffer(Block &b, core::IOBuffer &buffer, size_t size,
                                   size_t bufferOff, size_t blockOff)
{
    gmacError_t ret = gmacSuccess;
    const lazy::Block &block = dynamic_cast<const lazy::Block &>(b);
    switch(block.getState()) {
    case lazy::Invalid:
        ret = block.copyToBuffer(buffer, bufferOff, blockOff, size, lazy::Block::ACCELERATOR);
        break;
    case lazy::ReadOnly:
    case lazy::Dirty:
    case lazy::HostOnly:
        ret = block.copyToBuffer(buffer, bufferOff, blockOff, size, lazy::Block::HOST);
        break;
    }
    return ret;
}

gmacError_t LazyBase::copyFromBuffer(Block &b, core::IOBuffer &buffer, size_t size,
                                     size_t bufferOff, size_t blockOff)
{
    gmacError_t ret = gmacSuccess;
    lazy::Block &block = dynamic_cast<lazy::Block &>(b);
    switch(block.getState()) {
    case lazy::Invalid:
        ret = block.copyFromBuffer(blockOff, buffer, bufferOff, size, lazy::Block::ACCELERATOR);
        break;
    case lazy::ReadOnly:
#ifdef USE_OPENCL
        // WARNING: copying to host first because the IOBuffer address can change in copyToAccelerator
        // if we do not wait
        ret = block.copyFromBuffer(blockOff, buffer, bufferOff, size, lazy::Block::HOST);
        if(ret != gmacSuccess) break;
        ret = block.copyFromBuffer(blockOff, buffer, bufferOff, size, lazy::Block::ACCELERATOR);
        if(ret != gmacSuccess) break;
#else
        if(size == block.size() && blockOff == 0) { 
            if(block.protect(GMAC_PROT_NONE) < 0) FATAL("Unable to set memory permissions");
            block.setState(lazy::Invalid);

            ret = block.copyFromBuffer(blockOff, buffer, bufferOff, size, lazy::Block::ACCELERATOR);
            if(ret != gmacSuccess) break;
        }
        else {
            ret = block.copyFromBuffer(blockOff, buffer, bufferOff, size, lazy::Block::ACCELERATOR);
            if(ret != gmacSuccess) break;
            ret = block.copyFromBuffer(blockOff, buffer, bufferOff, size, lazy::Block::HOST);
            if(ret != gmacSuccess) break;
        }
#endif
        /* block.setState(lazy::Invalid); */
        break;
    case lazy::Dirty:
    case lazy::HostOnly:
        ret = block.copyFromBuffer(blockOff, buffer, bufferOff, size, lazy::Block::HOST);
        break;
    }
    return ret;
}

gmacError_t LazyBase::memset(const Block &b, int v, size_t size, size_t blockOffset)
{
    gmacError_t ret = gmacSuccess;
    const lazy::Block &block = dynamic_cast<const lazy::Block &>(b);
    switch(block.getState()) {
    case lazy::Invalid:
        ret = block.memset(v, size, blockOffset, lazy::Block::ACCELERATOR);
        break;
    case lazy::ReadOnly:
        ret = block.memset(v, size, blockOffset, lazy::Block::ACCELERATOR);
        if(ret != gmacSuccess) break;
        ret = block.memset(v, size, blockOffset, lazy::Block::HOST);
        break;
    case lazy::Dirty:
    case lazy::HostOnly:
        ret = block.memset(v, size, blockOffset, lazy::Block::HOST);
        break;
    }
    return ret;
}

#if 0
bool
LazyBase::isInAccelerator(Block &b)
{
    const lazy::Block &block = dynamic_cast<const lazy::Block &>(b);
    return block.getState() != lazy::Dirty;
}
#endif

gmacError_t
LazyBase::copyBlockToBlock(Block &d, size_t dstOffset, Block &s, size_t srcOffset, size_t count)
{
    lazy::Block &dst = dynamic_cast<lazy::Block &>(d);
    lazy::Block &src = dynamic_cast<lazy::Block &>(s);

    gmacError_t ret = gmacSuccess;

    if ((src.getState() == lazy::Invalid || src.getState() == lazy::ReadOnly) &&
        dst.getState() == lazy::Invalid) {
        TRACE(LOCAL, "I || R -> I");
        // Copy acc-acc
        ret = dst.copyFromBlock(dstOffset, src, srcOffset, count,
                                lazy::Block::ACCELERATOR,
                                lazy::Block::ACCELERATOR);
    } else if (src.getState() == lazy::Dirty && dst.getState() == lazy::Dirty) {
        // memcpy
        TRACE(LOCAL, "D -> D");
        ret = dst.copyFromBlock(dstOffset, src, srcOffset, count,
                                lazy::Block::HOST,
                                lazy::Block::HOST);
    } else if (src.getState() == lazy::ReadOnly && dst.getState() == lazy::ReadOnly) {
        TRACE(LOCAL, "R -> R");
        // Copy acc-to-acc
        // memcpy
        ret = dst.copyFromBlock(dstOffset, src, srcOffset, count,
                                lazy::Block::ACCELERATOR,
                                lazy::Block::ACCELERATOR);
        if (ret == gmacSuccess) {
            dst.copyFromBlock(dstOffset, src, srcOffset, count,
                              lazy::Block::HOST,
                              lazy::Block::HOST);
        }
    } else if (src.getState() == lazy::Invalid && dst.getState() == lazy::ReadOnly) {
        TRACE(LOCAL, "I -> R");
        // Copy acc-to-acc
        // acc-to-host
        ret = dst.copyFromBlock(dstOffset, src, srcOffset, count,
                                lazy::Block::ACCELERATOR,
                                lazy::Block::ACCELERATOR);
        if (ret == gmacSuccess) {
            ret = dst.copyFromBlock(dstOffset, src, srcOffset, count,
                                    lazy::Block::HOST,
                                    lazy::Block::ACCELERATOR);
        }
    } else if (src.getState() == lazy::Invalid && dst.getState() == lazy::Dirty) {
        TRACE(LOCAL, "I -> D");
        // acc-to-host
        ret = dst.copyFromBlock(dstOffset, src, srcOffset, count,
                                lazy::Block::HOST,
                                lazy::Block::ACCELERATOR);
    } else if (src.getState() == lazy::Dirty && dst.getState() == lazy::Invalid) {
        TRACE(LOCAL, "D -> I");
        // host-to-acc
        ret = dst.copyFromBlock(dstOffset, src, srcOffset, count,
                                lazy::Block::ACCELERATOR,
                                lazy::Block::HOST);
    } else if (src.getState() == lazy::Dirty && dst.getState() == lazy::ReadOnly) {
        // host-to-acc
        if (ret == gmacSuccess) {
            ret = dst.copyFromBlock(dstOffset, src, srcOffset, count,
                                    lazy::Block::ACCELERATOR,
                                    lazy::Block::HOST);
        }
        TRACE(LOCAL, "D -> R");
        // host-to-host
        ret = dst.copyFromBlock(dstOffset, src, srcOffset, count,
                          lazy::Block::HOST,
                          lazy::Block::HOST);
    } else if (src.getState() == lazy::ReadOnly && dst.getState() == lazy::Dirty) {
        TRACE(LOCAL, "R -> D");
        // host-to-host
        ret = dst.copyFromBlock(dstOffset, src, srcOffset, count,
                                lazy::Block::HOST,
                                lazy::Block::HOST);
    }

    TRACE(LOCAL, "Finished");
    return ret;
}

gmacError_t LazyBase::dump(Block &b, std::ostream &out, common::Statistic stat)
{
    lazy::BlockState &block = dynamic_cast<lazy::BlockState &>(b);
    //std::ostream *stream = (std::ostream *)param;
    //ASSERTION(stream != NULL);
    block.dump(out, stat);
    return gmacSuccess;
}

}}}
