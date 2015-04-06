/* Copyright (c) 2009, 2011 University of Illinois
                   Universitat Politecnica de Catalunya
                   All rights reserved.

Developed by: IMPACT Research Group / Grup de Sistemes Operatius
              University of Illinois / Universitat Politecnica de Catalunya
              http://impact.crhc.illinois.edu/
              http://gso.ac.upc.edu/

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal with the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
  1. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimers.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimers in the
     documentation and/or other materials provided with the distribution.
  3. Neither the names of IMPACT Research Group, Grup de Sistemes Operatius,
     University of Illinois, Universitat Politecnica de Catalunya, nor the
     names of its contributors may be used to endorse or promote products
     derived from this Software without specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
WITH THE SOFTWARE.  */

#ifndef GMAC_MEMORY_VM_BITMAP_H_
#define GMAC_MEMORY_VM_BITMAP_H_

#include "config/common.h"

#include "util/Lock.h"
#include "util/Logger.h"

#include "memory/Memory.h"

#ifdef USE_VM

#ifdef BITMAP_BYTE
#else
#ifdef BITMAP_BIT
#else
#error "ERROR: Bitmap granularity not defined!"
#endif
#endif

namespace __impl {

namespace core {
class Mode;
}

namespace memory  { namespace vm {

class Bitmap;

class GMAC_LOCAL Node
{
    friend class Bitmap;
private:
    unsigned level_;
    size_t nEntries_;
    size_t nUsedEntries_;
    std::vector<bool> usedEntries_;

    long_t firstUsedEntry_;
    long_t lastUsedEntry_;

    void syncToHost(long_t startIndex, long_t endIndex, size_t elemSize);
    void syncToAccelerator(long_t startIndex, long_t endIndex, size_t elemSize);
    void setDirty(bool synced);

    bool isDirty() const;

    void sync();

protected:
    long_t mask_;
    unsigned shift_;

    Bitmap &root_;
    hostptr_t entriesHost_;
    hostptr_t entriesAccHost_;
    accptr_t entriesAcc_;

    bool dirty_;
    bool synced_;

    long_t firstDirtyEntry_;
    long_t lastDirtyEntry_;

    std::vector<unsigned> nextEntries_;

    unsigned getLevel() const;

    long_t getLocalIndex(long_t index) const;
    long_t getGlobalIndex(long_t localIndex) const;
    long_t getNextIndex(long_t index) const;

    /**
     * Creates a child Node using the information of nextEntries_
     *
     * \return A child Node
     */
    Node *createChild() const;

    Node *getNode(long_t index);
    Node *&getNodeRef(long_t index);
    Node *&getNodeAccHostRef(long_t index);
    Node *getNodeAccAddr(long_t index);

    template <typename T>
    T getLeaf(long_t index);
    uint8_t &getLeafRef(long_t index);

    /**
     * Returns the index of the first modified entry
     *
     * \return The index of the first modified entry
     */
    long_t getFirstDirtyEntry() const;

    /**
     * Returns the index of the last modified entry
     *
     * \return The index of the last modified entry
     */
    long_t getLastDirtyEntry() const;

    /**
     * Update the indexes related to entry usage due new entries
     *
     * \param startIndex First entry
     * \param endIndex Last entry
     */
    void addEntries(long_t startIndex, long_t endIndex);

    /**
     * Update the indexes related to entry usage due entry removal
     *
     * \param startIndex First entry
     * \param endIndex Last entry
     */
    void removeEntries(long_t startIndex, long_t endIndex);

    /**
     * Update the indexes related to entry modification
     *
     * \param index Indef of the modified entry
     */
    void addDirtyEntry(long_t index);

    /**
     * Update the indexes related to entry modification
     *
     * \param startIndex Index of the first modified entry
     * \param endIndex Index of the last modified entry
     */
    void addDirtyEntries(long_t startIndex, long_t endIndex);

    /**
     * Allocates accelerator memory to be used by the Node
     *
     * \param isRoot A boolean that says if the Node is the root of the Bitmap
     */
    void allocAcc(bool isRoot);

    /**
     * Removes the accelerator memory used by the Node
     *
     * \param isRoot A boolean that says if the Node is the root of the Bitmap
     */
    void freeAcc(bool isRoot);

    /**
     * Synchronizes the contents of the Node from the accelerator memory
     * to the host memory
     *
     * \param startIndex The first index of the node to be synchronized
     * \param lastIndex The last index of the node to be synchronized
     */
    template <typename T>
    void syncToHost(long_t startIndex, long_t endIndex);

    /**
     * Synchronizes the contents of the Node from the host memory
     * to the accelerator memory
     *
     * \param startIndex The first index of the node to be synchronized
     * \param lastIndex The last index of the node to be synchronized
     */
    template <typename T>
    void syncToAccelerator(long_t startIndex, long_t endIndex);

    /**
     * Tells if the Node is synced between host and accelerator
     *
     * \return A boolean that tells if the Node is synced between host and
     * accelerator
     */
    bool isSynced() const;

    /**
     * Sets if the Node is synced between host and accelerator or not
     *
     * \param synced A boolean that tells if the Node is synced between host and
     * accelerator
     */
    void setSynced(bool synced);

    /**
     * Constructs a new Node
     *
     * \param level Level in the bitmap hierarchy
     * \param root Bitmap this Node belongs to
     * \param nEntries Number of entries in the node
     * \param nextEntries Number of entries in the next levels of the hierarchy
     */
    Node(unsigned level, Bitmap &root, size_t nEntries, std::vector<unsigned> nextEntries);

    /**
     * Destructs the Node
     */
    ~Node();
public:
    /**
     * Gets the state of the given entry
     *
     * \param addr Index of the entry
     * \return The state of the entry
     */
    template <typename T>
    T getEntry(long_t index);

    /**
     * Gets the state of the given entry and sets it to the given state
     *
     * \param addr Index of the entry
     * \param state State to be set for the entry
     * \return The state of the entry before being set
     */
    template <typename T>
    T getAndSetEntry(long_t index, T state);

    /**
     * Sets the state of the given entry
     *
     * \param index Index of the entry to be set
     * \param state State to be set
     */
    template <typename T>
    void setEntry(long_t index, T state);

    /**
     * Sets the state of the given entry range
     *
     * \param startIndex Index of the first entry to be set
     * \param endIndex Index of the last entry to be set
     * \param state State to be set
     */
    template <typename T>
    void setEntryRange(long_t startIndex, long_t endIndex, T state);

    /**
     * Tells if any of the entries contained in the given range
     * are in the given state
     *
     * \param startIndex Index of the first entry to be checked
     * \param endIndex Index of the last entry to be checked
     * \param state State of the block to be checked
     */
    template <typename T>
    bool isAnyInRange(long_t startIndex, long_t endIndex, T state);

    /**
     * Registers the given entry range to be managed by the Bitmap
     *
     * \param startIndex First entry
     * \param endIndex Last entry
     */
    void registerRange(long_t startIndex, long_t endIndex);

    /**
     * Unregisters the given entry range from be managed by the Bitmap
     *
     * \param startIndex First entry
     * \param endIndex Last entry
     */
    void unregisterRange(long_t startIndex, long_t endIndex);

    /**
     * Returns the number of used entries in the node
     *
     * \return The number of used entries in the node
     */
    size_t getNUsedEntries() const;

    /**
     * Returns the index of the first used entry in the node
     *
     * \return The index of the first used entry in the node
     */
    long_t getFirstUsedEntry() const;

    /**
     * Returns the index of the last used entry in the node
     *
     * \return The index of the last used entry in the node
     */
    long_t getLastUsedEntry() const;

    /**
     * Gets the ownership of the node
     */
    void acquire();

    /**
     * Releases the ownership of the node
     */
    void release();
};

class GMAC_LOCAL Bitmap
{
    friend class Node;

protected:
    /**
     * Number of levels of the bitmap
     */
    static const unsigned &BitmapLevels_;

    /**
     * Number of entries in the first level of the bitmap
     */
    static const unsigned &L1Entries_;

    /**
     * Number of entries in the second level of the bitmap
     */
    static const unsigned &L2Entries_;

    /**
     * Number of entries in the third level of the bitmap
     */
    static const unsigned &L3Entries_;

    /**
     * Mask used to get the index within the first level of the bitmap
     */
    static long_t L1Mask_;

    /**
     * Mask used to get the index within the second level of the bitmap
     */
    static long_t L2Mask_;

    /**
     * Mask used to get the index within the third level of the bitmap
     */
    static long_t L3Mask_;

    /**
     * Shift used to get the index within the first level of the bitmap
     */
    static unsigned L1Shift_;

    /**
     * Shift used to get the index within the second level of the bitmap
     */
    static unsigned L2Shift_;

    /**
     * Shift used to get the index within the third level of the bitmap
     */
    static unsigned L3Shift_;

    /**
     * Size in bytes of a block
     */
    static const size_t &BlockSize_;

    /**
     * Number of subblocks per block
     */
    static const unsigned &SubBlocks_;

    /**
     * Mode whose memory is managed by the bitmap
     */
    core::Mode &mode_;

    /**
     * Pointer to the first level (root level) of the bitmap
     */
    Node *root_;

    /**
     * Booleans that tells if the ownership of the bitmap has been released
     */
    bool released_;

    /**
     * Map of the registered memory ranges
     */
    std::map<accptr_t, size_t> ranges_;

    /**
     * Gets the entry index of the subblock containing the given address
     *
     * \param ptr Address of the subblock to retrieve the state from
     * \return The entry index of the subblock containing the given address
     */
    long_t getIndex(const accptr_t ptr) const;

    void syncToAccelerator();
public:
    /**
     * Constructs a new Bitmap
     */
    Bitmap(core::Mode &mode);

    /**
     * Initializes the common values needed for the Bitmap
     */
    static void Init();

    /**
     * Releases the resources used by the Bitmap
     */
    void cleanUp();

    /**
     * Sets the state of the subblock containing the given address
     *
     * \param addr Address of the subblock to be set
     * \param state State to be set to the block
     */
    template <typename T>
    void setEntry(const accptr_t addr, T state);

    /**
     * Sets the state of the subblocks within the given range
     *
     * \param addr Initial address of the subblocks to be set
     * \param bytes Size in bytes of the range
     * \param state State to be set to the blocks
     */
    template <typename T>
    void setEntryRange(const accptr_t addr, size_t bytes, T state);

    /**
     * Gets the state of the subblock containing the given address
     *
     * \param addr Address of the subblock to retrieve the state from
     * \return The state of the subblock
     */
    template <typename T>
    T getEntry(const accptr_t addr) const;

    /**
     * Gets the state of the subblock containing the given address, and
     * sets the state passed as a parameter
     *
     * \param addr Address of the subblock to retrieve the state from
     * \param state State to be set for the subblock
     * \return The state of the subblock
     */
    template <typename T>
    T getAndSetEntry(const accptr_t addr, T state);

    /**
     * Tells if any of the subblocks contained in the given range
     * are in the given state
     *
     * \param addr Initial address of the memory range
     * \param size Size in bytes of the memory range
     * \param state State of the block to be checked
     */
    template <typename T>
    bool isAnyInRange(const accptr_t addr, size_t size, T state);

    /**
     * Registers the given memory range to be managed by the Bitmap
     *
     * \param addr Initial address of the memory range
     * \param bytes Size in bytes of the memory range
     */
    void registerRange(const accptr_t addr, size_t bytes);

    /**
     * Unegisters the given memory range from the Bitmap
     *
     * \param addr Initial address of the memory range
     * \param bytes Size in bytes of the memory range
     */
    void unregisterRange(const accptr_t addr, size_t bytes);

    /**
     * Gets the ownership of the bitmap information
     */
    void acquire();

    /**
     * Releases the ownership of the bitmap information
     */
    void release();

    /**
     * Tells if the calling processing unit has the ownership of the
     * bitmap information 
     */
    bool isReleased() const;
};

}}}

#include "Model.h"

#include "Bitmap-impl.h"


#endif
#endif
