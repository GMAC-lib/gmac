#ifdef USE_VM

#include "api/cuda/hpe/Accelerator.h"
#include "api/cuda/hpe/Mode.h"
#include "api/cuda/hpe/Module.h"

#include "memory/vm/Bitmap.h"

namespace __impl { namespace memory { namespace vm {

static const char * ACC_VM_ROOT_VAR = "__gmac_vm_root";
#if 0
static const char * ACC_VM_SHIFT_L1 = "__gmac_vm_shift_l1";
static const char * ACC_VM_SHIFT_L2 = "__gmac_vm_shift_l2";
static const char * ACC_VM_SHIFT_L3 = "__gmac_vm_shift_l3";
static const char * ACC_VM_MASK_L1 = "__gmac_vm_mask_l1";
static const char * ACC_VM_MASK_L2 = "__gmac_vm_mask_l2";
static const char * ACC_VM_MASK_L3 = "__gmac_vm_mask_l3";
#endif

void
Node::allocAcc(bool isRoot)
{
    accptr_t addr(NULL);
    cuda::hpe::Mode &mode = reinterpret_cast<cuda::hpe::Mode &>(root_.mode_);

    if (isRoot == false) {
        gmacError_t ret = mode.map(addr, entriesAccHost_, nEntries_);
        ASSERTION(ret == gmacSuccess);
        TRACE(LOCAL,"Allocating a node in the accelerator. Size %zd. Addr %p", nEntries_, (void *)addr);
    } else {
        const cuda::hpe::Variable *var = mode.variableByName(ACC_VM_ROOT_VAR);
        ASSERTION(var != NULL);
        addr = var->devPtr();
        TRACE(LOCAL,"Using a node in the accelerator. Size %zd. Addr %p", nEntries_, (void *)addr);
    }
    entriesAcc_ = addr;
}

void
Node::freeAcc(bool isRoot)
{
    cuda::hpe::Mode &mode = reinterpret_cast<cuda::hpe::Mode &>(root_.mode_);
    if (isRoot == false) {
        gmacError_t ret = mode.unmap(entriesAccHost_, nEntries_);
        ASSERTION(ret == gmacSuccess);
        TRACE(LOCAL,"Freeing a node in the accelerator. Size %u. Addr %p", nEntries_, (void *)entriesAcc_);
    }
}


void
Node::syncToHost(long_t startIndex, long_t endIndex, size_t elemSize)
{
#ifndef USE_HOSTMAP_VM
    TRACE(LOCAL,"Syncing SharedBitmap");
    cuda::hpe::Mode &mode = reinterpret_cast<cuda::hpe::Mode &>(root_.mode_);
    gmacError_t ret;
    size_t size = (endIndex - startIndex + 1) * elemSize;
    hostptr_t host = entriesAccHost_ + startIndex * elemSize;
    accptr_t acc = entriesAcc_ + startIndex * elemSize;
    TRACE(LOCAL,"Setting dirty bitmap on host: %p -> %p: "FMT_SIZE, (void *) acc, host, size);
    ret = mode.copyToHost(host, acc, size);
    CFATAL(ret == gmacSuccess, "Unable to copy to host dirty bitmap node");
#endif
}

void
Node::syncToAccelerator(long_t startIndex, long_t endIndex, size_t elemSize)
{
    cuda::hpe::Mode &mode = reinterpret_cast<cuda::hpe::Mode &>(root_.mode_);
    cuda::hpe::Accelerator &acc = mode.getAccelerator();

#ifndef USE_HOSTMAP_VM
    if (isDirty()) {
        gmacError_t ret = gmacSuccess;
        accptr_t acc = entriesAcc_ + startIndex * elemSize;
        hostptr_t host = entriesAccHost_ + startIndex * elemSize;
        size_t size = (endIndex - startIndex + 1) * elemSize;

        ret = mode.copyToAccelerator(acc, host, size);
        TRACE(LOCAL,"Setting dirty bitmap on acc: %p -> %p: "FMT_SIZE, host, (void *) acc, size);
        CFATAL(ret == gmacSuccess, "Unable to copy dirty bitmap to accelerator");
        TRACE(LOCAL, "Syncing SharedBitmap");
        TRACE(LOCAL, "Copying "FMT_SIZE" bytes", size);

        setDirty(false);
    }
#endif
}


void
Bitmap::syncToAccelerator()
{
#ifndef USE_MULTI_CONTEXT
    cuda::hpe::Mode &mode = reinterpret_cast<cuda::hpe::Mode &>(mode_);
    cuda::hpe::Accelerator &acc = mode.getAccelerator();

    cuda::Mode *last = acc.getLastMode();

    if (last != &mode) {
        TRACE(LOCAL, "Syncing SharedBitmap pointers");
        gmacError_t ret = gmacSuccess;

#if 0
        const cuda::Variable *varShiftL1 = mode.constantByName(ACC_VM_SHIFT_L1);
        ASSERTION(varShiftL1 != NULL);
        accptr_t addrShiftL1 = varShiftL1->devPtr();

        const cuda::Variable *varShiftL2 = mode.constantByName(ACC_VM_SHIFT_L2);
        ASSERTION(varShiftL2 != NULL);
        accptr_t addrShiftL2 = varShiftL2->devPtr();

        const cuda::Variable *varShiftL3 = mode.constantByName(ACC_VM_SHIFT_L3);
        ASSERTION(varShiftL3 != NULL);
        accptr_t addrShiftL3 = varShiftL3->devPtr();

        const cuda::Variable *varMaskL1 = mode.constantByName(ACC_VM_MASK_L1);
        ASSERTION(varMaskL1 != NULL);
        accptr_t addrMaskL1 = varMaskL1->devPtr();

        const cuda::Variable *varMaskL2 = mode.constantByName(ACC_VM_MASK_L2);
        ASSERTION(varMaskL2 != NULL);
        accptr_t addrMaskL2 = varMaskL2->devPtr();

        const cuda::Variable *varMaskL3 = mode.constantByName(ACC_VM_MASK_L3);
        ASSERTION(varMaskL3 != NULL);
        accptr_t addrMaskL3 = varMaskL3->devPtr();


        ret = mode.copyToAccelerator(addrShiftL1, hostptr_t(&Bitmap::L1Shift_), sizeof(Bitmap::L1Shift_));
        CFATAL(ret == gmacSuccess, "Unable to set the number of L1 entries in the accelerator %p", (void *) addrShiftL1);
        ret = mode.copyToAccelerator(addrShiftL2, hostptr_t(&Bitmap::L2Shift_), sizeof(Bitmap::L2Shift_));
        CFATAL(ret == gmacSuccess, "Unable to set the number of L2 entries in the accelerator %p", (void *) addrShiftL2);
        ret = mode.copyToAccelerator(addrShiftL3, hostptr_t(&Bitmap::L3Shift_), sizeof(Bitmap::L3Shift_));
        CFATAL(ret == gmacSuccess, "Unable to set the number of L3 entries in the accelerator %p", (void *) addrShiftL3);

        ret = mode.copyToAccelerator(addrMaskL1, hostptr_t(&Bitmap::L1Mask_), sizeof(Bitmap::L1Mask_));
        CFATAL(ret == gmacSuccess, "Unable to set the number of L1 entries in the accelerator %p", (void *) addrMaskL1);
        ret = mode.copyToAccelerator(addrMaskL2, hostptr_t(&Bitmap::L2Mask_), sizeof(Bitmap::L2Mask_));
        CFATAL(ret == gmacSuccess, "Unable to set the number of L2 entries in the accelerator %p", (void *) addrMaskL2);
        ret = mode.copyToAccelerator(addrMaskL3, hostptr_t(&Bitmap::L3Mask_), sizeof(Bitmap::L3Mask_));
        CFATAL(ret == gmacSuccess, "Unable to set the number of L3 entries in the accelerator %p", (void *) addrMaskL3);
#endif
    }

    acc.setLastMode(mode);
#endif
}


#if 0
void SharedBitmap::allocate()
{
    ASSERTION(accelerator_ == NULL);
    cuda::Mode &mode = static_cast<cuda::Mode &>(mode_);
#ifdef USE_HOSTMAP_VM
    mode.hostAlloc((void **)&bitmap_, size_);
    accelerator_ = (uint8_t *) mode.hostMap(bitmap_);
    memset(bitmap_, 0, size());
    TRACE(LOCAL,"Allocating dirty bitmap (%zu bytes)", size());
#else
    mode.malloc(&accelerator_, size_);
    TRACE(LOCAL,"Allocating dirty bitmap %p -> %p (%zu bytes)", bitmap_, (void *) accelerator_, size_);
#endif
}


void SharedBitmap::allocate()
{
    ASSERTION(accelerator_ == NULL);
    cuda::Mode &mode = static_cast<cuda::Mode &>(mode_);
#ifdef USE_HOSTMAP_VM
    mode.hostAlloc((void **)&bitmap_, size_);
    accelerator_ = (uint8_t *) mode.hostMap(bitmap_);
    memset(bitmap_, 0, size());
    TRACE(LOCAL,"Allocating dirty bitmap (%zu bytes)", size());
#else
    mode.malloc(&accelerator_, size_);
    TRACE(LOCAL,"Allocating dirty bitmap %p -> %p (%zu bytes)", bitmap_, (void *) accelerator_, size_);
#endif
}

void
SharedBitmap::cleanUp()
{
    cuda::Mode &mode = static_cast<cuda::Mode &>(mode_);
    if(accelerator_ != NULL) mode.hostFree(bitmap_);
#ifndef USE_HOSTMAP_VM
    if(!linked_) {
        Bitmap::cleanUp();
    }
#endif
}

void
SharedBitmap::syncHost()
{
#ifndef USE_HOSTMAP_VM
    TRACE(LOCAL,"Syncing SharedBitmap");
    cuda::Mode &mode = static_cast<cuda::Mode &>(mode_);
    TRACE(LOCAL,"Setting dirty bitmap on host: %p -> %p: "FMT_SIZE, (void *) accelerator(), host(), size());
    gmacError_t ret;
    //printf("SharedBitmap toHost\n");
    ret = mode.copyToHost(host(), accelerator(), size());
    CFATAL(ret == gmacSuccess, "Unable to copy back dirty bitmap");
    reset();
#endif
}

void
SharedBitmap::syncAccelerator()
{
    if (accelerator_ == NULL) allocate();

    cuda::Mode &mode = static_cast<cuda::Mode &>(mode_);
    cuda::Accelerator &acc = dynamic_cast<cuda::Accelerator &>(mode.getAccelerator());

#ifndef USE_MULTI_CONTEXT
    cuda::Mode *last = acc.getLastMode();

    if (last != &mode_) {
        if (last != NULL) {
            SharedBitmap &lastBitmap = last->acceleratorDirtyBitmap();
            if (!lastBitmap.synced()) {
                lastBitmap.syncHost();
            }
        }
        TRACE(LOCAL, "Syncing SharedBitmap pointers");
        TRACE(LOCAL, "%p -> %p (0x%lx)", host(), (void *) accelerator(), mode.dirtyBitmapAccPtr());
        gmacError_t ret = gmacSuccess;
        accptr_t bitmapAccPtr = mode.dirtyBitmapAccPtr();
        accptr_t bitmapShiftPageAccPtr = mode.dirtyBitmapShiftPageAccPtr();
        ret = mode.copyToAccelerator(bitmapAccPtr, hostptr_t(&accelerator_.ptr_), sizeof(accelerator_.ptr_));
        CFATAL(ret == gmacSuccess, "Unable to set the pointer in the accelerator %p", (void *) mode.dirtyBitmapAccPtr());
        ret = mode.copyToAccelerator(bitmapShiftPageAccPtr, hostptr_t(&shiftPage_), sizeof(shiftPage_));
        CFATAL(ret == gmacSuccess, "Unable to set shift page in the accelerator %p", (void *) mode.dirtyBitmapShiftPageAccPtr());
    }

#ifndef USE_HOSTMAP_VM
    if (dirty_) {
        TRACE(LOCAL, "Syncing SharedBitmap");
        TRACE(LOCAL, "Copying "FMT_SIZE" bytes. ShiftPage: %d", size(), shiftPage_);
        gmacError_t ret = gmacSuccess;
        ret = mode.copyToAccelerator(accelerator(), host(), size());
        CFATAL(ret == gmacSuccess, "Unable to copy dirty bitmap to accelerator");
    }

    synced_ = false;
#endif

#ifndef USE_MULTI_CONTEXT
    acc.setLastMode(mode);
#endif
#endif
}
#endif

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
