/* Copyright (c) 2009, 2010, 2011 University of Illinois
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

#ifndef GMAC_CORE_HPE_PROCESS_H_
#define GMAC_CORE_HPE_PROCESS_H_

#include <list>
#include <map>
#include <vector>

#include "config/common.h"
#include "config/order.h"
#include "include/gmac/types.h"
#include "memory/ObjectMap.h"

#include "util/Private.h"
#include "util/UniquePtr.h"

#include "core/Process.h"

#include "Queue.h"

namespace __impl { namespace core { namespace hpe {

class Accelerator;

/** Map that contains in which accelerator resides a mode */
class GMAC_LOCAL ModeMap : private std::map<Mode *, unsigned>, gmac::util::Lock {
    friend class Process;
private:
    typedef std::map<Mode *, unsigned> Parent;

public:
    /** Constructs the ModeMap */
    ModeMap();

    typedef Parent::iterator iterator;
    typedef Parent::const_iterator const_iterator;

    /**
     * Inserts a mode/accelerator pair in the map
     * \param mode Mode to be inserted
     * \return A pair that contains the position where the items have been
     * allocated and a boolean that tells if the items have been actually
     * inserted
     */
    std::pair<iterator, bool> insert(Mode *mode);
    void remove(Mode &mode);
};

class GMAC_LOCAL QueueMap :
    private std::map<THREAD_T, ThreadQueue *>, gmac::util::RWLock
{
private:
    typedef std::map<THREAD_T, ThreadQueue *> Parent;
public:
    QueueMap();

    typedef Parent::iterator iterator;

    void cleanup();
    std::pair<iterator, bool> insert(THREAD_T, ThreadQueue *);
    void push(THREAD_T id, Mode &mode);
    Mode *pop();
    void erase(THREAD_T id);
};

/** Represents the resources used by a running process */
class GMAC_LOCAL Process : public core::Process, public gmac::util::RWLock {
    DBC_FORCE_TEST(Process)
protected:
    typedef Accelerator *AcceleratorPtr;
    std::vector<AcceleratorPtr> accs_;

    std::vector<AddressSpace *> aSpaces_;

    ModeMap modes_;
    memory::Protocol &protocol_;
    QueueMap queues_;
    memory::ObjectMap shared_;
    memory::ObjectMap global_;
    memory::ObjectMap orphans_;

    unsigned current_;

    /**
     * Destroys the process and releases the resources used by it
     */
    virtual ~Process();

public:
    /**
     * Constructs the process
     */
    Process();

    /**
     * Registers a new thread in the process
     */
    TESTABLE void initThread();

    /**
     * Unregisters a thread from the process
     */
    TESTABLE void finiThread();

#define ACC_AUTO_BIND -1
    /**
     * Creates a new Mode in the process for the calling thread and binds it to
     * the given accelerator
     *
     * \param acc The accelerator id which the mode will be bound to or
     * ACC_AUTO_BIND to let the run-time choose
     * \return A pointer to the newly created mode or NULL if there has been an error
     */
     TESTABLE Mode *createMode(int acc = ACC_AUTO_BIND);

    /**
     * Removes a mode from the process
     *
     * \param mode A reference to the mode to be removed from the process
     */
     TESTABLE void removeMode(Mode &mode);

     /**
      * Registers a global object in the process
      *
      * \param object Reference to the object to be registered
      * \return Error code
     */
    gmacError_t globalMalloc(memory::Object &object);

    /**
     * Unregisters a global object from the process
     *
     * \param object Reference to the object to be unregistered
     * \return Error code
     */
    gmacError_t globalFree(memory::Object &object);

    /**
     * Translates a host address to an accelerator address
     *
     * \param addr Host address to be translated
     * \return Accelerator address
     */
    accptr_t translate(const hostptr_t addr);

    /*****************************/
    /* Mode management functions */
    /*****************************/

    /**
     * The calling thread sends the ownership of its mode to the given thread
     *
     * \param id Identifer of the thread that receives the ownership of the mode
     */
    void send(THREAD_T id);

    /**
     * The calling thread receives the ownership of a mode sent by another thread
     */
    void receive();

    /**
     * The calling thread sends the ownership of its mode to the given thread
     * and receives the ownership of the mode sent by another thread
     *
     * \param id Identifer of the thread that receives the ownership of the mode
     */
    void sendReceive(THREAD_T id);

    /**
     * The calling thread shares the ownership of its mode with the given thread
     *
     * \param id Identifer of the thread that shares the ownership of the mode
     * with the calling thread
     */
    void copy(THREAD_T id);

    /**
     * Migrates the given mode to the given accelerator
     *
     * \param acc Identifier of the destination accelerator
     * \return Error code
     */
    gmacError_t migrate(int acc);

    /**
     * Adds an accelerator to the process so it can be used by the threads of the
     * process
     *
     * \param acc A reference to the accelerator to be added
     */
    void addAccelerator(Accelerator &acc);

    /**
     * Gets the number of accelerators available in the process
     *
     * \return The number of accelerators available in the process
     */
    size_t nAccelerators() const;

    /**
     * Gets the accelerator with the given id
     *
     * \param i An accelerator id
     * \return A reference to the accelerator with the given id, or NULL if a non-valid id is given
     */
    Accelerator &getAccelerator(unsigned i);


    /**
     * Tells if all the available accelerators in the process are integrated and
     * therefore share the physical memory with the CPU
     *
     * \return A boolean that tells if all the available accelerators in the
     * process are integrated
     */
    bool allIntegrated();

    /**
     * Gets the protocol used by the process for the global objects
     *
     * \return A reference to the protocol used by the process for the global
     * objects
     */
    memory::Protocol *getProtocol();

    /**
     * Gets the object map that contains all the shared objects allocated in the
     * process
     *
     * \return A reference to the object map that contains all the shared
     * objects allocated in the process
     */
    memory::ObjectMap &shared();

    /**
     * Gets the object map that contains all the shared objects allocated in the
     * process
     *
     * \return A constant reference to the object map that contains all the shared
     * objects allocated in the process
     */
    const memory::ObjectMap &shared() const;

    /**
     * Gets the object map that contains all the global objects allocated in the
     * process
     *
     * \return A reference to the object map that contains all the global
     * objects allocated in the process
     */
    memory::ObjectMap &global();

    /**
     * Gets the object map that contains all the global objects allocated in the
     * process
     *
     * \return A constant reference to the object map that contains all the
     * global objects allocated in the process
     */
    const memory::ObjectMap &global() const;

    /**
     * Gets the object map that contains all the objects that have been orphaned
     * in the process
     *
     * \return A reference to the object map that contains all the objects that
     * have been orphaned in the process
     */
    memory::ObjectMap &orphans();

    /**
     * Inserts an object into the orphan (objects without owner) list
     * \param object Object that becomes orphan
     */
    void makeOrphan(memory::Object &object);

    /**
     * Gets the object map that contains all the objects that have been orphaned
     * in the process
     *
     * \return A constant reference to the object map that contains all the objects that
     * have been orphaned in the process
     */
    const memory::ObjectMap &orphans() const;

    /**
     * Returns the owner of the object with the smallest address within the
     * given memory range
     *
     * \param addr Starting address of the range
     * \param size Size of the range
     * \return The owner of the object with the smallest address within the
     * given memory range
     */
    core::Mode *owner(const hostptr_t addr, size_t size = 0);

    /**
     * Waits for pending operations before a kernel call (needed for distributed objects)
     *
     * \return gmacSuccess on success, an error code otherwise
     */
    gmacError_t prepareForCall();

    gmacError_t setAddressSpace(Mode &mode, unsigned aSpaceId);
};

}}}

#include "Process-impl.h"

#ifdef USE_DBC
#include "core/hpe/dbc/Process.h"
#endif


#include "core/hpe/Mode.h"

#endif
