#ifdef USE_VM

#include <cmath>
#include <cstring>

#include "core/Mode.h"
#include "memory/Memory.h"

#include "Bitmap.h"

namespace __impl { namespace memory { namespace vm {

const unsigned &Bitmap::BitmapLevels_ = util::params::ParamBitmapLevels;
const unsigned &Bitmap::L1Entries_ = util::params::ParamBitmapL1Entries;
const unsigned &Bitmap::L2Entries_ = util::params::ParamBitmapL2Entries;
const unsigned &Bitmap::L3Entries_ = util::params::ParamBitmapL3Entries;
const size_t &Bitmap::BlockSize_ = util::params::ParamBlockSize;
const unsigned &Bitmap::SubBlocks_ = util::params::ParamSubBlocks;

long_t Bitmap::L1Mask_;
long_t Bitmap::L2Mask_;
long_t Bitmap::L3Mask_;
unsigned Bitmap::L1Shift_;
unsigned Bitmap::L2Shift_;
unsigned Bitmap::L3Shift_;

Node::Node(unsigned level, Bitmap &root, size_t nEntries, std::vector<unsigned> nextEntries) :
    level_(level), nEntries_(nEntries), nUsedEntries_(0),
    usedEntries_(nEntries),
    firstUsedEntry_(-1), lastUsedEntry_(-1),
    root_(root),
    entriesAcc_(NULL),
    dirty_(false),
    synced_(true),
    nextEntries_(nextEntries)
{
    TRACE(LOCAL, "Node constructor");

    long shift = 0;
    for (size_t i = 0; i < nextEntries_.size(); i++) {
        shift += log2(nextEntries[i]);
    }

    for (size_t i = 0; i < nEntries; i++) {
        usedEntries_[i] = false;
    }

    mask_  = (long_t(nEntries) - 1) << shift;
    shift_ = shift;
    TRACE(LOCAL, "Entries: %zd", nEntries_);
    TRACE(LOCAL, "Shift: %u", shift_);
    TRACE(LOCAL, "Mask : %lx", mask_);

    if (nextEntries_.size() > 0) {
        entriesHost_ = hostptr_t(::malloc(nEntries * sizeof(uint8_t)));
        ::memset(entriesHost_, 0, nEntries * sizeof(uint8_t));
    } else {
        entriesHost_ = NULL;
    }
    entriesAccHost_ = hostptr_t(::malloc(nEntries * sizeof(uint8_t)));
    TRACE(LOCAL, "Allocating memory: %p", entriesHost_);
    ::memset(entriesAccHost_, 0, nEntries * sizeof(uint8_t));
}

Node::~Node()
{
    freeAcc(getLevel() == 0);

    TRACE(LOCAL, "NodeStore destructor");

    if (nextEntries_.size() > 0) {
        for (long_t i = getFirstUsedEntry(); i <= getLastUsedEntry(); i++) {
            Node *node = getNode(i);
            if (node != NULL) {
                delete node;
            }
        }
    }

    if (entriesHost_ != NULL) {
        ::free(entriesHost_);
    }
    ::free(entriesAccHost_);
}

void
Node::registerRange(long_t startIndex, long_t endIndex)
{
    long_t localStartIndex = getLocalIndex(startIndex);
    long_t localEndIndex = getLocalIndex(endIndex);

    TRACE(LOCAL, "registerRange 0x%lx 0x%lx", localStartIndex, localEndIndex);

    addEntries(localStartIndex, localEndIndex);
    addDirtyEntries(localStartIndex, localEndIndex);

    if (entriesAcc_ == NULL) {
        allocAcc(getLevel() == 0);
        TRACE(LOCAL, "Allocating accelerator memory: %p", entriesAcc_.get());
    }

    if (nextEntries_.size() == 0) {
        for (long_t i = localStartIndex; i <= localEndIndex; i++) {
            uint8_t &leaf = getLeafRef(i);
            leaf = uint8_t(0);
        }
        return;
    }

    long_t startWIndex = startIndex;
    long_t endWIndex   = (localStartIndex == localEndIndex)? endIndex:
                                getGlobalIndex(localStartIndex + 1) - 1;

    long_t i = localStartIndex;
    do {
        Node *&node = getNodeRef(i);
        if (node == NULL) {
            node = static_cast<Node *>(createChild());
            node->allocAcc(false);
            TRACE(LOCAL, "Allocating accelerator memory: %p", node->entriesAcc_.get());

            Node *&nodeAcc = getNodeAccHostRef(i);
            nodeAcc = node->getNodeAccAddr(0);
            TRACE(LOCAL, "linking with 0x%p", nodeAcc);
        }

        node->registerRange(getNextIndex(startWIndex), getNextIndex(endWIndex));
        i++;
        startWIndex = getGlobalIndex(i);
        endWIndex = (i < localEndIndex)? getGlobalIndex(i + 1) - 1: endIndex;
    } while (i <= localEndIndex);
}

void
Node::unregisterRange(long_t startIndex, long_t endIndex)
{
    long_t localStartIndex = getLocalIndex(startIndex);
    long_t localEndIndex = getLocalIndex(endIndex);

    TRACE(LOCAL, "unregisterRange 0x%lx 0x%lx", localStartIndex, localEndIndex);

    if (nextEntries_.size() == 0) {
        removeEntries(localStartIndex, localEndIndex);
        return;
    }

    long_t startWIndex = startIndex;
    long_t endWIndex   = (localStartIndex == localEndIndex)? endIndex:
                                getGlobalIndex(localStartIndex + 1) - 1;

    long_t i = localStartIndex;
    do {
        Node *&node = getNodeRef(i);
        node->unregisterRange(getNextIndex(startWIndex), getNextIndex(endWIndex));
        if (node->getNUsedEntries() == 0) {
            Node *&nodeAcc = getNodeAccHostRef(i);
            delete node;
            node = NULL;
            nodeAcc = NULL;
        }
        i++;
        startWIndex = getGlobalIndex(i);
        endWIndex = (i < localEndIndex)? getGlobalIndex(i + 1) - 1: endIndex;
    } while (i <= localEndIndex);
}


Node *
Node::createChild() const
{
    if (nextEntries_.size() == 0) return NULL;

    std::vector<unsigned> nextEntries(nextEntries_.size() - 1);
    std::copy(++nextEntries_.begin(), nextEntries_.end(), nextEntries.begin());
    Node *node = new Node(getLevel() + 1, root_, nextEntries_[0], nextEntries);
    return node;
}

void
Node::acquire()
{
    TRACE(LOCAL, "Acquire");

    if (nextEntries_.size() > 0) {
        for (long_t i = getFirstUsedEntry(); i <= getLastUsedEntry(); i++) {
            Node *node = static_cast<Node *>(getNode(i));
            if (node != NULL) node->acquire();
        }
    } else {
        // Only leaf nodes must be synced
        this->setSynced(false);
    }
}

void
Node::release()
{
    if (nextEntries_.size() > 0) {
        for (long_t i = getFirstUsedEntry(); i <= getLastUsedEntry(); i++) {
            Node *node = static_cast<Node *>(getNode(i));
            if (node != NULL) node->release();
        }
    }

    if (this->isDirty()) {
        if (nextEntries_.size() > 0) {
            this->syncToAccelerator<Node *>(this->firstDirtyEntry_, this->lastDirtyEntry_);
        } else {
            this->syncToAccelerator<uint8_t>(this->firstDirtyEntry_, this->lastDirtyEntry_);
        }
    }
}

void
Bitmap::Init()
{
    unsigned shift = SubBlockShift_;

    if (BitmapLevels_ == 3) {
        Bitmap::L3Shift_ = shift;
        Bitmap::L3Mask_  = long_t(Bitmap::L3Entries_ - 1) << shift;
        shift += log2(Bitmap::L3Entries_);
        TRACE(GLOBAL, "L3SEntries_ %u", Bitmap::L3Shift_);
        TRACE(GLOBAL, "L3Shift %u", Bitmap::L3Shift_);
        TRACE(GLOBAL, "L3Mask %lu", Bitmap::L3Mask_);
    }

    Bitmap::L2Shift_ = shift;
    Bitmap::L2Mask_  = (long_t(Bitmap::L2Entries_) - 1) << shift;
    shift += log2(Bitmap::L2Entries_);
    TRACE(GLOBAL, "L2Entries %u", Bitmap::L2Entries_);
    TRACE(GLOBAL, "L2Shift %u", Bitmap::L2Shift_);
    TRACE(GLOBAL, "L2Mask %lu", Bitmap::L2Mask_);

    Bitmap::L1Shift_ = shift;
    Bitmap::L1Mask_  = (long_t(Bitmap::L1Entries_) - 1) << shift;
    TRACE(GLOBAL, "L1Shift %u", Bitmap::L1Shift_);
    TRACE(GLOBAL, "L1Mask %lu", Bitmap::L1Mask_);
}

Bitmap::Bitmap(core::Mode &mode) :
    mode_(mode),
    released_(false)
{
    TRACE(LOCAL, "Bitmap constructor");

    std::vector<unsigned> nextEntries;

    if (BitmapLevels_ > 1) {
        nextEntries.push_back(L2Entries_);
    }
    if (BitmapLevels_ == 3) {
        nextEntries.push_back(L3Entries_);
    }

    root_ = new Node(0, *this, L1Entries_, nextEntries);
}

void
Bitmap::cleanUp()
{
    delete root_;
}

void
Bitmap::registerRange(const accptr_t addr, size_t bytes)
{
    TRACE(LOCAL, "registerRange %p %zd", (void *) addr, bytes);

    root_->registerRange(getIndex(addr), getIndex(addr + bytes - 1));
}

void
Bitmap::unregisterRange(const accptr_t addr, size_t bytes)
{
    TRACE(LOCAL, "unregisterRange %p %zd", (void *) addr, bytes);

    root_->unregisterRange(getIndex(addr), getIndex(addr + bytes - 1));
}

#if 0
#ifdef BITMAP_BIT
const unsigned Bitmap::EntriesPerByte_ = 8;
#else // BITMAP_BYTE
const unsigned Bitmap::EntriesPerByte_ = 1;
#endif

Bitmap::Bitmap(core::Mode &mode, unsigned bits) :
    RWLock("Bitmap"), bits_(bits), mode_(mode), bitmap_(NULL), dirty_(true), minPtr_(NULL), maxPtr_(NULL)
{
    unsigned rootEntries = (1 << bits) >> 32;
    if (rootEntries == 0) rootEntries = 1;
    rootEntries_ = rootEntries;

    bitmap_ = new hostptr_t[rootEntries];
    ::memset(bitmap_, 0, rootEntries * sizeof(hostptr_t));

    shiftBlock_ = int(log2(util::params::ParamPageSize));
    shiftPage_  = shiftBlock_ - int(log2(util::params::ParamSubBlocks));

    subBlockSize_ = (util::params::ParamSubBlocks) - 1;
    subBlockMask_ = (util::params::ParamSubBlocks) - 1;
    pageMask_     = subBlockSize_ - 1;

    size_    = (1 << (bits - shiftPage_)) / EntriesPerByte_;
#ifdef BITMAP_BIT
    bitMask_ = (1 << 3) - 1;
#endif

    TRACE(LOCAL, "Pages: %u", 1 << (bits - shiftPage_));
    TRACE(LOCAL,"Size : %u", size_);
}

Bitmap::Bitmap(const Bitmap &base) :
    RWLock("Bitmap"),
    bits_(base.bits_),
    mode_(base.mode_),
    bitmap_(base.bitmap_),
    dirty_(true),
    shiftBlock_(base.shiftBlock_),
    shiftPage_(base.shiftPage_),
    subBlockSize_(base.subBlockSize_),
    subBlockMask_(base.subBlockMask_),
    pageMask_(base.pageMask_),
#ifdef BITMAP_BIT
    bitMask_(base.bitMask_),
#endif
    size_(base.size_),
    minEntry_(-1), maxEntry_(-1)
{
}


Bitmap::~Bitmap()
{
    
}

void
Bitmap::cleanUp()
{
    for (long int i = minRootEntry_; i <= maxRootEntry_; i++) {
        if (bitmap_[i] != NULL) {
            delete [] bitmap_[i];
        }
    }
    delete [] bitmap_;
}

SharedBitmap::SharedBitmap(core::Mode &mode, unsigned bits) :
    Bitmap(mode, bits), linked_(false), synced_(true), accelerator_(NULL)
{
}

SharedBitmap::SharedBitmap(const Bitmap &host) :
    Bitmap(host), linked_(true), synced_(true), accelerator_(NULL)
{
}

SharedBitmap::~SharedBitmap()
{
}


#ifdef DEBUG_BITMAP
void Bitmap:dump()
{
    core::Context * ctx = Mode::current()->context();
    ctx->invalidate();

    static int idx = 0;
    char path[256];
    sprintf(path, "_bitmap__%d", idx++);
    FILE * file = fopen(path, "w");
    fwrite(bitmap_, 1, size_, file);    
    fclose(file);
}
#endif
#endif

}}}

#endif
