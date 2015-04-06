#if 0
#ifdef USE_VM

#include "Gather.h"

#include "core/IOBuffer.h"

#include "config/config.h"

#include "memory/Memory.h"
#include "memory/StateBlock.h"

#include "trace/Tracer.h"


#if defined(__GNUC__)
#define MIN std::min
#elif defined(_MSC_VER)
#define MIN min
#endif

namespace __impl { namespace memory { namespace protocol {


GatherBase::GatherBase(size_t limit) :
    gmac::util::Lock("Gather"),
    limit_(limit)
{
}

GatherBase::~GatherBase()
{

}

GatherBase::State GatherBase::state(GmacProtection prot) const
{
	switch(prot) {
		case GMAC_PROT_NONE: 
			return Invalid;
		case GMAC_PROT_READ:
			return ReadOnly;
		case GMAC_PROT_WRITE:
		case GMAC_PROT_READWRITE:
			return Dirty;
	}
	return Dirty;
}


void GatherBase::deleteObject(Object &obj)
{
    obj.release();
}



bool GatherBase::needUpdate(const Block &b) const
{
    const StateBlock<State> &block = dynamic_cast<const StateBlock<State> &>(b);
    switch(block.state()) {        
        case Dirty:
        case HostOnly:
            return false;
        case ReadOnly:
        case Invalid:
            return true;
    }
    return false;
}

gmacError_t GatherBase::signalRead(Block &b, hostptr_t addr)
{
    trace::EnterCurrentFunction();
	StateBlock<State> &block = dynamic_cast<StateBlock<State> &>(b);
    gmacError_t ret = gmacSuccess;

    if(block.state() == HostOnly) {
        WARNING("Signal on HostOnly block - Changing protection and continuing");
        Memory::protect(block.addr(), block.size(), GMAC_PROT_READWRITE);
        goto exit_func;
    }

    if (block.state() != Invalid) {
        goto exit_func; // Somebody already fixed it
    }

    ret = block.toHost();
    if(ret != gmacSuccess) goto exit_func;
    Memory::protect(block.addr(), block.size(), GMAC_PROT_READ);
    block.state(ReadOnly);

exit_func:
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t GatherBase::signalWrite(Block &b, hostptr_t addr)
{
    trace::EnterCurrentFunction();
    StateBlock<State> &block = dynamic_cast<StateBlock<State> &>(b);
    gmacError_t ret = gmacSuccess;

    hostptr_t start;
    size_t count;

    switch(block.state()) {
        case Dirty:            
            if (!block.isSequentialAccess()) {
                start = block.getSubBlockAddr(addr);
                count = block.getSubBlockSize();

                block.setSubBlockDirty(addr);
            } else {
                start = block.addr();
                count = block.size();

                block.setBlockDirty();
            }
            Memory::protect(start, count, GMAC_PROT_READWRITE);
            goto exit_func; // Somebody already fixed it
        case Invalid:          
            ret = block.toHost();
            if(ret != gmacSuccess) goto exit_func;
            if (!block.isSequentialAccess()) {
                start = block.getSubBlockAddr(addr);
                count = block.getSubBlockSize();

                block.setSubBlockDirty(addr);
            } else {
                start = block.addr();
                count = block.size();

                block.setBlockDirty();
            }
			Memory::protect(start, count, GMAC_PROT_READWRITE);
            break;
        case HostOnly:
            WARNING("Signal on HostOnly block - Changing protection and continuing");
        case ReadOnly:
            if (!block.isSequentialAccess()) {
                start = block.getSubBlockAddr(addr);
                count = block.getSubBlockSize();

                block.setSubBlockDirty(addr);
            } else {
                start = block.addr();
                count = block.size();

                block.setBlockDirty();
            }
			Memory::protect(start, count, GMAC_PROT_READWRITE);
            break;
    }
    block.state(Dirty);
    if (float(b.getFaults()) <
        float(b.getSubBlocks()) * util::params::ParamGatherRatio) {
        //ret = block.toGatherBuffer();
        buffers_.addSubBlock(block, block.getSubBlock(addr));
    } else {
        buffers_.removeBlock(block);
        addDirty(block);
    }
    TRACE(LOCAL,"Setting block %p to dirty state", start);
exit_func:
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t GatherBase::acquire(Block &b)
{
    gmacError_t ret = gmacSuccess;
    StateBlock<State> &block = dynamic_cast<StateBlock<State> &>(b);
    switch(block.state()) {
        case Invalid:
        case ReadOnly:
			if(Memory::protect(block.addr(), block.size(), GMAC_PROT_NONE) < 0)
                FATAL("Unable to set memory permissions");
            break;
        case Dirty:
            FATAL("Block in incongruent state in acquire: %p", block.addr());
            break;
        case HostOnly:
            break;
    }
	return ret;
}

gmacError_t GatherBase::acquireWithBitmap(Block &b)
{
    gmacError_t ret = gmacSuccess;
    // TODO: Get mode as parameter
    core::Mode &mode = core::Mode::getCurrent();
    vm::BitmapShared &acceleratorBitmap = mode.acceleratorDirtyBitmap();
    StateBlock<State> &block = dynamic_cast<StateBlock<State> &>(b);
    switch(block.state()) {
        case Invalid:
        case ReadOnly:
            if (acceleratorBitmap.isAnyInRange(block.acceleratorAddr(block.addr()), block.size(), vm::BITMAP_SET_ACC)) {
                if(Memory::protect(block.addr(), block.size(), GMAC_PROT_NONE) < 0)
                    FATAL("Unable to set memory permissions");
                block.state(Invalid);
            } else {
                if(Memory::protect(block.addr(), block.size(), GMAC_PROT_READ) < 0)
                    FATAL("Unable to set memory permissions");
                block.state(ReadOnly);
            }
            break;
        case Dirty:
            FATAL("Block in incongruent state in acquire: %p", block.addr());
            break;
        case HostOnly:
            break;
    }
	return ret;
}

gmacError_t GatherBase::mapToAccelerator(Block &b)
{
    memory::StateBlock<State> &block = dynamic_cast<memory::StateBlock<State> &>(b);
    ASSERTION(block.state() == HostOnly);
    TRACE(LOCAL,"Mapping block to accelerator %p", block.addr());
    block.state(Dirty);
    block.setBlockDirty();
    addDirty(block);
    return gmacSuccess;
}

gmacError_t GatherBase::unmapFromAccelerator(Block &b)
{
    memory::StateBlock<State> &block = dynamic_cast<memory::StateBlock<State> &>(b);
    TRACE(LOCAL,"Unmapping block from accelerator %p", block.addr());
    gmacError_t ret = gmacSuccess;
    switch(block.state()) {
        case HostOnly:
        case Dirty:
        case ReadOnly:
            break;
        case Invalid:
            ret = block.toHost();
            if(ret != gmacSuccess) break;
    }
    if(Memory::protect(block.addr(), block.size(), GMAC_PROT_READWRITE) < 0)
        FATAL("Unable to set memory permissions");
    block.state(HostOnly);
    dbl_.remove(block);
    return ret;
}

void GatherBase::addDirty(Block &block)
{
    dbl_.push(block);
    if(limit_ == size_t(-1)) return;
    while(dbl_.size() > limit_) {
        Block *b = dbl_.pop();
        b->coherenceOp(&Protocol::release);
    }
    return;
}

gmacError_t GatherBase::releaseObjects()
{
    // We need to make sure that this operations is done before we
    // let other modes to proceed
    lock(); 
    while(dbl_.empty() == false) {
        Block *b = dbl_.pop();
        b->coherenceOp(&Protocol::release);
    }
    unlock();
    return gmacSuccess;
}

template<typename T>
inline gmacError_t SharedBlock<T>::toGatherBuffer(Block &b, GatherBufferCache &buffers)
{
    gmacError_t ret = gmacSuccess;
    // TODO: get mode as parameter
    core::Mode &mode = core::Mode::getCurrent();
#ifdef USE_SUBBLOCK_TRACKING
    vm::BitmapHost &bitmap   = mode.acceleratorDirtyBitmap();
#else
#ifdef USE_VM
    vm::BitmapShared &bitmap = mode.acceleratorDirtyBitmap();
#endif
#endif
    //fprintf(stderr, "TODEVICE: SubBlocks %u\n", Block::getSubBlocks());
    for (unsigned i = 0; i < b.getSubBlocks(); i++) {
        if (bitmap.getAndSetEntry(acceleratorAddr_ + i * SubBlockSize_,
                                  vm::BITMAP_UNSET) == vm::BITMAP_SET_HOST) {
            buffers.addSubBlock(b, i);
        }
    }
	return true;
}


gmacError_t GatherBase::release(Block &b)
{
    StateBlock<State> &block = dynamic_cast<StateBlock<State> &>(b);
    TRACE(LOCAL,"Releasing block %p", block.addr());
    gmacError_t ret = gmacSuccess;
    switch(block.state()) {
        case Dirty:
            if (float(b.getFaults()) <
                float(b.getSubBlocks()) * util::params::ParamGatherRatio) {
                //ret = block.toGatherBuffer();
            } else {
                ret = block.toAccelerator();
                if(ret != gmacSuccess) break;
                if(Memory::protect(block.addr(), block.size(), GMAC_PROT_READ) < 0)
                    FATAL("Unable to set memory permissions");
                block.state(ReadOnly);
            }
            break;
        case Invalid:
        case ReadOnly:
        case HostOnly:
            break;
    }
    return ret;
}

gmacError_t GatherBase::deleteBlock(Block &block)
{
    dbl_.remove(dynamic_cast<StateBlock<State> &>(block));
    return gmacSuccess;
}

gmacError_t GatherBase::toHost(Block &b)
{
    TRACE(LOCAL,"Sending block to host: %p", b.addr());
    gmacError_t ret = gmacSuccess;
    StateBlock<State> &block = dynamic_cast<StateBlock<State> &>(b);
    switch(block.state()) {
        case Invalid:
            TRACE(LOCAL,"Invalid block");
			if(Memory::protect(block.addr(), block.size(), GMAC_PROT_READ) < 0)
                FATAL("Unable to set memory permissions");
            ret = block.toHost();
            if(ret != gmacSuccess) break;
            block.state(ReadOnly);
            break;
        case Dirty:
            TRACE(LOCAL,"Dirty block");
            break;
        case ReadOnly:
            TRACE(LOCAL,"ReadOnly block");
            break;
        case HostOnly:
            TRACE(LOCAL,"HostOnly block");
            break;
    }
    return ret;
}

gmacError_t GatherBase::toAccelerator(Block &b)
{
    TRACE(LOCAL,"Sending block to accelerator: %p", b.addr());
    gmacError_t ret = gmacSuccess;
    StateBlock<State> &block = dynamic_cast<StateBlock<State> &>(b);
    switch(block.state()) {
        case Dirty:
            TRACE(LOCAL,"Dirty block");
            ret = block.toAccelerator();
            if(ret != gmacSuccess) break;
            if(Memory::protect(block.addr(), block.size(), GMAC_PROT_READ) < 0)
                FATAL("Unable to set memory permissions");
            block.state(ReadOnly);
            break;
        case Invalid:
            TRACE(LOCAL,"Invalid block");
            break;
        case ReadOnly:
            TRACE(LOCAL,"ReadOnly block");
            break;
        case HostOnly:
            TRACE(LOCAL,"HostOnly block");
            break;
    }
    return ret;
}

gmacError_t GatherBase::copyToBuffer(const Block &b, core::IOBuffer &buffer, size_t size,
							   size_t bufferOffset, size_t blockOffset) const
{
	gmacError_t ret = gmacSuccess;
	const StateBlock<State> &block = dynamic_cast<const StateBlock<State> &>(b);
	switch(block.state()) {
		case Invalid:
			ret = block.copyFromAccelerator(buffer, size, bufferOffset, blockOffset);
			break;
		case ReadOnly:
		case Dirty:
        case HostOnly:
			ret = block.copyFromHost(buffer, size, bufferOffset, blockOffset);
	}
	return ret;
}

gmacError_t GatherBase::copyFromBuffer(const Block &b, core::IOBuffer &buffer, size_t size, 
							   size_t bufferOffset, size_t blockOffset) const
{
	gmacError_t ret = gmacSuccess;
	const StateBlock<State> &block = dynamic_cast<const StateBlock<State> &>(b);
	switch(block.state()) {
		case Invalid:
			ret = block.copyToAccelerator(buffer, size, bufferOffset, blockOffset);
			break;
		case ReadOnly:
			ret = block.copyToAccelerator(buffer, size, bufferOffset, blockOffset);
			if(ret != gmacSuccess) break;
			ret = block.copyToHost(buffer, size, bufferOffset, blockOffset);
			break;
		case Dirty:			
        case HostOnly:
			ret = block.copyToHost(buffer, size, bufferOffset, blockOffset);
			break;
	}
	return ret;
}

gmacError_t GatherBase::memset(const Block &b, int v, size_t size, size_t blockOffset) const
{
    gmacError_t ret = gmacSuccess;
	const StateBlock<State> &block = dynamic_cast<const StateBlock<State> &>(b);
	switch(block.state()) {
		case Invalid:
            ret = b.acceleratorMemset(v, size, blockOffset);
			break;
		case ReadOnly:
			ret = b.acceleratorMemset(v, size, blockOffset);
			if(ret != gmacSuccess) break;
			ret = b.hostMemset(v, size, blockOffset);
			break;
		case Dirty:			
        case HostOnly:
			ret = b.hostMemset(v, size, blockOffset);
			break;
	}
	return ret;
}

}}}

#endif
#endif
