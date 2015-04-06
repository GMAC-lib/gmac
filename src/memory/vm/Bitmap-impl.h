#ifndef GMAC_MEMORY_VM_BITMAP_IMPL_H_
#define GMAC_MEMORY_VM_BITMAP_IMPL_H_

namespace __impl { namespace memory { namespace vm {

inline
unsigned
Node::getLevel() const
{
    return level_;
}

inline
size_t
Node::getNUsedEntries() const
{
    return nUsedEntries_;
}

inline
long_t
Node::getFirstUsedEntry() const
{
    return firstUsedEntry_;
}

inline
long_t
Node::getLastUsedEntry() const
{
    return lastUsedEntry_;
}

inline
void
Node::addEntries(long_t startIndex, long_t endIndex)
{
    if (nUsedEntries_ == 0) {
        firstUsedEntry_ = startIndex;
        lastUsedEntry_ = endIndex;
    } else {
        if (firstUsedEntry_ > startIndex) firstUsedEntry_ = startIndex;
        if (lastUsedEntry_ < endIndex) lastUsedEntry_ = endIndex;

    }
    if (lastUsedEntry_ >= nEntries_) FATAL("STH IS WROOOONG %lu", lastUsedEntry_);

    for (long_t i = startIndex; i <= endIndex; i++) {
        if (usedEntries_[i] == false) {
            usedEntries_[i] = true;
            nUsedEntries_++;
        }
    }
}

inline
void
Node::removeEntries(long_t startIndex, long_t endIndex)
{
    for (long_t i = startIndex; i <= endIndex; i++) {
        if (usedEntries_[i] == true) {
            usedEntries_[i] = false;
            nUsedEntries_--;
        }
    }

    if (nUsedEntries_ > 0) {
        bool first = false;
        for (long_t i = 0; i < nEntries_; i++) {
            if (first == false && usedEntries_[i] == true) {
                firstUsedEntry_ = i;
                first = true;
            }

            if (first == true && usedEntries_[i] == true) {
                lastUsedEntry_ = i;
            }
        }
    } else {
        firstUsedEntry_ = -1;
        lastUsedEntry_ = -1;
    }
}

inline
bool
Node::isSynced() const
{
    return synced_;
}

inline
void
Node::setSynced(bool synced)
{
    synced_ = synced;
}

/*
inline
accptr_t
Node::getAccAddr() const
{
    return entriesAcc_;
}
*/

inline bool
Node::isDirty() const
{
    return dirty_;
}

inline
void
Node::setDirty(bool dirty)
{
    dirty_ = dirty;
}

inline
void
Node::addDirtyEntry(long_t index)
{
    if (isDirty() == false) {
        firstDirtyEntry_ = index;
        lastDirtyEntry_ = index;
        setDirty(true);
    } else {
        if (firstDirtyEntry_ > index) firstDirtyEntry_ = index;
        if (lastDirtyEntry_ < index) lastDirtyEntry_ = index;
    }
}

inline
void
Node::addDirtyEntries(long_t startIndex, long_t endIndex)
{
    if (isDirty() == false) {
        firstDirtyEntry_ = startIndex;
        lastDirtyEntry_ = endIndex;
        setDirty(true);
    } else {
        if (firstDirtyEntry_ > startIndex) firstDirtyEntry_ = startIndex;
        if (lastDirtyEntry_ < endIndex) lastDirtyEntry_ = endIndex;
    }
}

#if 0
inline
Node::StoreShared(Bitmap &root, size_t size, bool allocHost) :
    dirty_(false),
    synced_(true)
{
    TRACE(LOCAL, "StoreShared constructor");

    }

StoreShared::~StoreShared()
{
    TRACE(LOCAL, "StoreShared destructor");
    }
#endif

template <typename T>
inline void
Node::syncToHost(long_t startIndex, long_t endIndex)
{
    syncToHost(startIndex, endIndex, sizeof(T));
}

template <typename T>
inline void
Node::syncToAccelerator(long_t startIndex, long_t endIndex)
{
    syncToAccelerator(startIndex, endIndex, sizeof(T));
}

inline
long_t
Node::getFirstDirtyEntry() const
{
    return firstDirtyEntry_;
}

inline
long_t
Node::getLastDirtyEntry() const
{
    return lastDirtyEntry_;
}

inline
long_t
Node::getLocalIndex(long_t index) const
{
    TRACE(LOCAL, "getLocalIndex (%lx & %lx) >> %u -> %lx", index, mask_, shift_, (index & mask_) >> shift_);
    long_t ret = (index & mask_) >> shift_;
    return ret;
}

inline
long_t
Node::getGlobalIndex(long_t localIndex) const
{
    return localIndex << shift_;
}

inline
long_t
Node::getNextIndex(long_t index) const
{
    TRACE(LOCAL, "getNextIndex %lx ~ %lx-> %lx", index, ~mask_, index & ~mask_);
    return index & ~mask_;
}

#if 0
Node::NodeStore(unsigned level, Bitmap &root, size_t nEntries, std::vector<unsigned> nextEntries) :
    Node(level, nEntries, nextEntries),
    NodeShared(root, nEntries * (nextEntries.size() == 0? sizeof(uint8_t):
                                                          sizeof(Node *)),
    nextEntries.size() > 0)
{
    TRACE(LOCAL, "NodeStore constructor");
}
#endif

inline Node *
Node::getNode(long_t index)
{
    return reinterpret_cast<Node **>(this->entriesHost_)[index];
}

inline Node *&
Node::getNodeRef(long_t index)
{
    return reinterpret_cast<Node **>(this->entriesHost_)[index];
}

template <typename T>
inline
T
Node::getLeaf(long_t index)
{
    uint8_t val = reinterpret_cast<uint8_t *>(this->entriesAccHost_)[index];
    return T(val);
}

inline
uint8_t &
Node::getLeafRef(long_t index)
{
    return static_cast<uint8_t *>(this->entriesAccHost_)[index];
}

inline
Node *&
Node::getNodeAccHostRef(long_t index)
{
    return reinterpret_cast<Node **>(this->entriesAccHost_)[index];
}

inline
Node *
Node::getNodeAccAddr(long_t index)
{
    return static_cast<Node *>(reinterpret_cast<Node **>((void *) this->entriesAcc_) + index);
}

#if 0
inline
NodeShared::NodeShared(unsigned level, Bitmap &root, size_t nEntries, std::vector<unsigned> nextEntries) :
    NodeStore<StoreShared>(level, root, nEntries, nextEntries)
{
    TRACE(LOCAL, "NodeShared constructor");
}
#endif

template <typename T>
T
Node::getEntry(long_t index)
{
    sync();

    long_t localIndex = this->getLocalIndex(index);

    TRACE(LOCAL, "getEntry 0x%lx", localIndex);
    if (this->nextEntries_.size() == 0) {
        return getLeaf<T>(localIndex);
    } else {
        long_t nextIndex = this->getNextIndex(index);
        Node *node = getNode(localIndex);
        return node->getEntry<T>(nextIndex);
    }
}

template <typename T>
T
Node::getAndSetEntry(long_t index, T state)
{
    sync();

    long_t localIndex = getLocalIndex(index);
    addDirtyEntry(localIndex);

    TRACE(LOCAL, "getAndSetEntry 0x%lx", localIndex);
    if (this->nextEntries_.size() == 0) {
        uint8_t &ref = getLeafRef(localIndex);
        T val = T(ref);
        ref = state;
        return val;
    } else {
        long_t nextIndex = this->getNextIndex(index);
        Node *node = getNode(localIndex);
        return node->getAndSetEntry(nextIndex, state);
    }
}

template <typename T>
void
Node::setEntry(long_t index, T state)
{
    sync();

    long_t localIndex = getLocalIndex(index);
    TRACE(LOCAL, "setEntry 0x%lx", localIndex);
    addDirtyEntry(localIndex);

    TRACE(LOCAL, "setEntry 0x%lx", localIndex);
    if (this->nextEntries_.size() == 0) {
        uint8_t &ref = getLeafRef(localIndex);
        ref = state;
    } else {
        long_t nextIndex = this->getNextIndex(index);
        Node *node = getNode(localIndex);
        ASSERTION(node != NULL);
        node->setEntry(nextIndex, state);
    }
}

template <typename T>
void
Node::setEntryRange(long_t startIndex, long_t endIndex, T state)
{
    sync();

    long_t localStartIndex = this->getLocalIndex(startIndex);
    long_t localEndIndex = this->getLocalIndex(endIndex);
 
    addDirtyEntries(localStartIndex, localEndIndex);

    TRACE(LOCAL, "setEntryRange 0x%lx 0x%lx", localStartIndex, localEndIndex);
    if (this->nextEntries_.size() == 0) {
        for (long_t i = localStartIndex; i <= localEndIndex; i++) {
            uint8_t &leaf = getLeafRef(i);
            leaf = state;
        }
        return;
    }

    long_t startWIndex = startIndex;
    long_t endWIndex   = (localStartIndex == localEndIndex)? endIndex:
                                this->getGlobalIndex(localStartIndex + 1) - 1;

    long_t i = localStartIndex;
    do {
        Node *node = getNode(i);
        node->setEntryRange(this->getNextIndex(startWIndex), this->getNextIndex(endWIndex), state);
        i++;
        startWIndex = this->getGlobalIndex(i);
        endWIndex = (i < localEndIndex)? this->getGlobalIndex(i + 1) - 1: endIndex;
    } while (i <= localEndIndex);
}

template <typename T>
bool
Node::isAnyInRange(long_t startIndex, long_t endIndex, T state)
{
    sync();

    long_t localStartIndex = this->getLocalIndex(startIndex);
    long_t localEndIndex = this->getLocalIndex(endIndex);
 
    TRACE(LOCAL, "isAnyInRange 0x%lx 0x%lx", localStartIndex, localEndIndex);
    if (this->nextEntries_.size() == 0) {
        for (long_t i = localStartIndex; i <= localEndIndex; i++) {
            if (getLeaf<T>(i) == state) return true;
        }
        return false;
    }

    long_t startWIndex = startIndex;
    long_t endWIndex   = (localStartIndex == localEndIndex)? endIndex:
                                this->getGlobalIndex(localStartIndex + 1) - 1;

    long_t i = localStartIndex;
    do {
        Node *node = getNode(i);
        bool ret = node->isAnyInRange(this->getNextIndex(startWIndex), this->getNextIndex(endWIndex), state);
        if (ret == true) return true;
        i++;
        startWIndex = this->getGlobalIndex(i);
        endWIndex = (i < localEndIndex)? this->getGlobalIndex(i + 1) - 1: endIndex;
    } while (i <= localEndIndex);

    return false;
}


inline
void
Node::sync()
{
    TRACE(LOCAL, "sync");

    if (!this->isSynced()) {
        if (nextEntries_.size() > 0) {
            this->syncToHost<Node *>(getFirstUsedEntry(), getLastUsedEntry());
        } else {
            this->syncToHost<uint8_t>(getFirstUsedEntry(), getLastUsedEntry());
        }
        this->setSynced(true);
    }
}

template <typename T>
inline
void
Bitmap::setEntry(const accptr_t addr, T state)
{
    TRACE(LOCAL, "setEntry %p", (void *) addr);

    long_t entry = getIndex(addr);
    root_->setEntry<T>(entry, state);
}

template <typename T>
inline
void
Bitmap::setEntryRange(const accptr_t addr, size_t bytes, T state)
{
    TRACE(LOCAL, "setEntryRange %p %zd", (void *) addr, bytes);

    long_t firstEntry = getIndex(addr);
    long_t lastEntry = getIndex(addr + bytes - 1);
    root_->setEntryRange<T>(firstEntry, lastEntry, state);
}

inline
long_t
Bitmap::getIndex(const accptr_t _ptr) const
{
    void * ptr = (void *) _ptr;
    long_t index = long_t(ptr);
    index >>= memory::SubBlockShift_;
    return index;
}

template <typename T>
inline
T
Bitmap::getEntry(const accptr_t addr) const
{
    TRACE(LOCAL, "getEntry %p", (void *) addr);
    long_t entry = getIndex(addr);
    T state = root_->getEntry<T>(entry);
    TRACE(LOCAL, "getEntry ret: %d", state);
    return state;
}

template <typename T>
inline
T
Bitmap::getAndSetEntry(const accptr_t addr, T state)
{
    TRACE(LOCAL, "getAndSetEntry %p", (void *) addr);
    long_t entry = getIndex(addr);
    T ret= root_->getAndSetEntry<T>(entry, state);
    TRACE(LOCAL, "getAndSetEntry ret: %d", ret);
    return ret;
}


template <typename T>
inline
bool
Bitmap::isAnyInRange(const accptr_t addr, size_t size, T state)
{
    TRACE(LOCAL, "isAnyInRange %p %zd", (void *) addr, size);

    long_t firstEntry = getIndex(addr);
    long_t lastEntry = getIndex(addr + size - 1);
    return root_->isAnyInRange<T>(firstEntry, lastEntry, state);
}

inline void
Bitmap::acquire()
{
    TRACE(LOCAL, "Acquire");

    if (released_ == true) {
        TRACE(LOCAL, "Acquiring");
        root_->acquire();
        released_ = false;
    }
}

inline void
Bitmap::release()
{
    TRACE(LOCAL, "Release");

    if (released_ == false) {
        TRACE(LOCAL, "Releasing");
        // Sync the device variables
        syncToAccelerator();

        // Sync the bitmap contents
        root_->release();

        released_ = true;
    }
}

inline bool
Bitmap::isReleased() const
{
    return released_;
}

}}}

#endif
