#include "config/order.h"

#include "libs/common.h"

#include "util/Atomics.h"
#include "util/Logger.h"
#include "util/Private.h"

static __impl::util::Private<bool> inGmac_;
__impl::util::Private<bool> isRunTimeThread_;

static Atomic gmacInit__ = 0;
static Atomic gmacCtor_ = 0;

static volatile bool gmacIsInitialized = false;

const bool privateTrue = true;
const bool privateFalse = false;

CONSTRUCTOR(init);
static void init(void)
{
    if(AtomicTestAndSet(gmacCtor_, 0, 1) == 1) return;
    /* Create GMAC enter lock and set GMAC as initialized */
    __impl::util::Private<bool>::init(inGmac_);
    __impl::util::Private<bool>::init(isRunTimeThread_);

    inGmac_.set(&privateFalse);
    isRunTimeThread_.set(&privateFalse);
#ifdef POSIX
    threadInit();
#endif
}

void enterGmac()
{
    if(AtomicTestAndSet(gmacInit__, 0, 1) == 0) {
        inGmac_.set(&privateTrue);
        initGmac();
        gmacIsInitialized = true;
    } else if (*isRunTimeThread_.get() == privateFalse) {
        while (!gmacIsInitialized);
        inGmac_.set(&privateTrue);
    } else {
        inGmac_.set(&privateTrue);
    }
}


void enterGmacExclusive()
{
    if (AtomicTestAndSet(gmacInit__, 0, 1) == 0) initGmac();
    inGmac_.set(&privateTrue);
}

void exitGmac()
{
    inGmac_.set(&privateFalse);
}

bool inGmac()
{
    bool *ret = inGmac_.get();
    ASSERTION(ret != NULL);
    return *ret;
}
