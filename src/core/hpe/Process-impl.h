#ifndef GMAC_CORE_HPE_PROCESS_IMPL_H_
#define GMAC_CORE_HPE_PROCESS_IMPL_H_

#include "AddressSpace.h"

namespace __impl { namespace core { namespace hpe {

inline size_t
Process::nAccelerators() const
{
    return accs_.size();
}

inline Accelerator &
Process::getAccelerator(unsigned i)
{
    ASSERTION(i < accs_.size(), "Incorrect accelerator ID");

    return *accs_[i];
}

inline memory::Protocol *
Process::getProtocol()
{
    return &protocol_;
}

inline memory::ObjectMap &
Process::shared()
{
    return shared_;
}

inline const memory::ObjectMap &
Process::shared() const
{
    return shared_;
}

inline memory::ObjectMap &
Process::global()
{
    return global_;
}

inline const memory::ObjectMap &
Process::global() const
{
    return global_;
}

inline memory::ObjectMap &
Process::orphans()
{
    return orphans_;
}

inline const memory::ObjectMap &
Process::orphans() const
{
    return orphans_;
}

inline void
Process::makeOrphan(memory::Object &obj)
{
    TRACE(LOCAL, "Making orphan object: %p", obj.addr());
    // Insert into the orphan list
    orphans_.addObject(obj);
    // Remove from the list of regular shared objects
    shared_.removeObject(obj);
}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
