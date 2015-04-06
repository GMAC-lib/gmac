/*
 * <+ DESCRIPTION +>
 *
 * Copyright (C) 2011, Javier Cabezas <jcabezas in ac upc edu> {{{
 *
 * This program is free software; you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License 
 * as published by the Free Software Foundation; either 
 * version 2 of the License, or any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 * }}}
 */

#ifndef GMAC_MEMORY_VM_MODEL_H_
#define GMAC_MEMORY_VM_MODEL_H_

#if defined(USE_VM) || defined(USE_SUBBLOCK_TRACKING)

namespace __impl {

namespace memory  { namespace vm {

enum ModelDirection {
    MODEL_TOHOST = 0,
    MODEL_TODEVICE = 1
};

template <ModelDirection M>
static inline
float costTransferCache(const size_t subBlockSize, size_t subBlocks)
{
    if (M == MODEL_TOHOST) {
        if (subBlocks * subBlockSize <= util::params::ParamModelL1/2) {
            return util::params::ParamModelToHostTransferL1;
        } else if (subBlocks * subBlockSize <= util::params::ParamModelL2/2) {
            return util::params::ParamModelToHostTransferL2;
        } else {
            return util::params::ParamModelToHostTransferMem;
        }
    } else {
        if (subBlocks * subBlockSize <= util::params::ParamModelL1/2) {
            return util::params::ParamModelToDeviceTransferL1;
        } else if (subBlocks * subBlockSize <= util::params::ParamModelL2/2) {
            return util::params::ParamModelToDeviceTransferL2;
        } else {
            return util::params::ParamModelToDeviceTransferMem;
        }
    }
}

template <ModelDirection M>
static inline
float costGaps(const size_t subBlockSize, unsigned gaps, unsigned subBlocks)
{
    return costTransferCache<M>(subBlockSize, subBlocks) * gaps * subBlockSize;
}

template <ModelDirection M>
static inline
float costTransfer(const size_t subBlockSize, size_t subBlocks)
{
    return costTransferCache<M>(subBlockSize, subBlocks) * subBlocks * subBlockSize;
}

template <ModelDirection M>
static inline
float costConfig()
{
    if (M == MODEL_TOHOST) {
        return util::params::ParamModelToHostConfig;
    } else {
        return util::params::ParamModelToDeviceConfig;
    }
}

template <ModelDirection M>
static inline
float cost(const size_t subBlockSize, size_t subBlocks)
{
    return costConfig<M>() + costTransfer<M>(subBlockSize, subBlocks);
}

}}}

#endif

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
