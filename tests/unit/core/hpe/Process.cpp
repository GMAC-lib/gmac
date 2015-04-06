#include "unit/core/hpe/Process.h"

#include "core/hpe/Mode.h"
#include "core/hpe/Process.h"
#include "core/hpe/Thread.h"
#include "memory/ObjectMap.h"


using __impl::core::hpe::Accelerator;
using __impl::core::hpe::Mode;
using __impl::core::hpe::ModeMap;

using gmac::core::hpe::Process;
using gmac::core::hpe::Thread;

using __impl::memory::Object;
using __impl::memory::Protocol;

extern void CUDA(Process &);
extern void OpenCL(Process &);

Process *ProcessTest::createProcess()
{
    Process *proc = new Process();
    if(proc == NULL) return proc;
#if defined(USE_CUDA)
    CUDA(*proc);
#endif
#if defined(USE_OPENCL)
    OpenCL(*proc);
#endif
    __impl::memory::Init();
    return proc;
}

TEST_F(ProcessTest, ModeMap) {
    Process *proc = createProcess();
    ASSERT_TRUE(proc != NULL);

    Mode *mode = proc->createMode(0);
    ASSERT_TRUE(mode != NULL);

    ModeMap mm;
    typedef std::map<__impl::core::hpe::Mode *, unsigned> Parent;
    typedef Parent::iterator iterator;
    std::pair<iterator, bool> ib = mm.insert(mode);
    ASSERT_TRUE(ib.second);
    mm.remove(*mode);

    proc->destroy();
}

TEST_F(ProcessTest, GlobalMemory) {
    Process *proc = createProcess();

    const size_t size = 4 * 1024 * 1024;
    Protocol *protocol = proc->getProtocol();
    ASSERT_TRUE(protocol != NULL);
    Object *object = protocol->createObject(Thread::getCurrentMode(), size, NULL, GMAC_PROT_READ, 0);
    ASSERT_TRUE(object != NULL);
    ASSERT_EQ(gmacSuccess, proc->globalMalloc(*object));
    ASSERT_TRUE(object->addr() != NULL);
    ASSERT_TRUE(proc->translate(object->addr()) != accptr_t(0));
    ASSERT_EQ(gmacSuccess, proc->globalFree(*object));
    ASSERT_TRUE(proc->translate(object->addr()) == accptr_t(0));

    object->decRef();
    proc->destroy();
}

TEST_F(ProcessTest, Mode) {
	size_t count;
	Process *proc = createProcess();
	count = proc->nAccelerators();
	ASSERT_EQ(gmacSuccess, proc->migrate(static_cast<int>(count) - 1));
	Mode &mode = Thread::getCurrentMode();
	Accelerator &acc = mode.getAccelerator();
	ASSERT_TRUE(acc.id() == count-1);

	ASSERT_FALSE(proc->allIntegrated());
	
	proc->destroy();
}