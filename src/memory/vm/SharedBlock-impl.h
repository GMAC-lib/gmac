#ifndef GMAC_MEMORY_VM_SHAREDBLOCK_IMPL_H_
#define GMAC_MEMORY_VM_SHAREDBLOCK_IMPL_H_
#if 0

template<typename T>
inline gmacError_t SharedBlock<T>::toHost() const
{
    gmacError_t ret = gmacSuccess;

    vm::BitmapShared &acceleratorBitmap = owner_.acceleratorDirtyBitmap();
    bool inSubGroup = false;
    unsigned groupStart = 0, groupEnd = 0;
    unsigned gaps = 0;

    for (unsigned i = 0; i < Block::getSubBlocks(); i++) {
        if (inSubGroup) {
            if (acceleratorBitmap.getAndSetEntry(acceleratorAddr_ + i * SubBlockSize_, vm::BITMAP_UNSET) == vm::BITMAP_SET_ACC) {
                groupEnd = i;
            } else {
                if (vm::costGaps<vm::MODEL_TODEVICE>(SubBlockSize_, gaps + 1, i - groupStart + 1) <
                    vm::cost<vm::MODEL_TODEVICE>(SubBlockSize_, 1)) {
                    gaps++;
                } else {
                    inSubGroup = false;

                    ret = owner_.copyToHost(StateBlock<T>::shadow_ + groupStart * SubBlockSize_,
                                            acceleratorAddr_       + groupStart * SubBlockSize_,
                                            (groupEnd - groupStart + 1) * SubBlockSize_);
                    if (ret != gmacSuccess) break;
                }
            }
        } else {
            if (acceleratorBitmap.getAndSetEntry(acceleratorAddr_ + i * SubBlockSize_, vm::BITMAP_UNSET) == vm::BITMAP_SET_ACC) {
                groupStart = groupEnd = i; gaps = 0; inSubGroup = true;
            }
        }
    }
    if (inSubGroup) {
        ret = owner_.copyToHost(StateBlock<T>::shadow_ + groupStart * SubBlockSize_,
                                acceleratorAddr_       + groupStart * SubBlockSize_,
                                (groupEnd - groupStart + 1) * SubBlockSize_);
    }
    return ret;
}

template<typename T>
inline gmacError_t SharedBlock<T>::toAccelerator()
{
    gmacError_t ret = gmacSuccess;
    vm::BitmapShared &bitmap= owner_.acceleratorDirtyBitmap();

    bool inSubGroup = false;
    unsigned groupStart = 0, groupEnd = 0;
    unsigned gaps = 0;

    for (unsigned i = 0; i < Block::getSubBlocks(); i++) {
        if (inSubGroup) {
            if (bitmap.getAndSetEntry(acceleratorAddr_ + i * SubBlockSize_, vm::BITMAP_UNSET) == vm::BITMAP_SET_HOST) {
                groupEnd = i;
            } else {
                if (vm::costGaps<vm::MODEL_TODEVICE>(SubBlockSize_, gaps + 1, i - groupStart + 1) <
                    vm::cost<vm::MODEL_TODEVICE>(SubBlockSize_, 1)) {
                    gaps++;
                } else {
                    inSubGroup = false;
                    
                    ret = owner_.copyToAccelerator(acceleratorAddr_       + groupStart * SubBlockSize_,
                                                   StateBlock<T>::shadow_ + groupStart * SubBlockSize_,
                                                   (groupEnd - groupStart + 1) * SubBlockSize_);
                    if (ret != gmacSuccess) break;
                }
            }
        } else {
            if (bitmap.getAndSetEntry(acceleratorAddr_ + i * SubBlockSize_, vm::BITMAP_UNSET) == vm::BITMAP_SET_HOST) {
                groupStart = groupEnd = i; gaps = 0; inSubGroup = true;
            }
        }
    }
    if (inSubGroup) {
        ret = owner_.copyToAccelerator(acceleratorAddr_       + groupStart * SubBlockSize_,
                                       StateBlock<T>::shadow_ + groupStart * SubBlockSize_,
                                       (groupEnd - groupStart + 1) * SubBlockSize_);
    }
    Block::resetBitmapStats();
	return ret;
}
#endif

#endif
