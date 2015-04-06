/* Copyright (c) 2009, 2010 University of Illinois
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

#ifndef GMAC_API_CUDA_HPE_MODULE_H_
#define GMAC_API_CUDA_HPE_MODULE_H_

#include <cuda.h>
#include <driver_types.h>
#include <texture_types.h>

#include <list>
#include <map>
#include <string>
#include <vector>

#include "config/common.h"
#include "config/config.h"

#include "util/UniquePtr.h"

#include "Kernel.h"



namespace __impl { namespace cuda { namespace hpe {

typedef const char *gmacVariable_t;
typedef const struct textureReference *gmacTexture_t;

typedef core::hpe::Descriptor<gmacTexture_t> TextureDescriptor;

class GMAC_LOCAL VariableDescriptor : public core::hpe::Descriptor<gmacVariable_t> {
protected:
    bool constant_;

public:
    VariableDescriptor(const char *name, gmacVariable_t key, bool constant);
    bool constant() const;
};

class GMAC_LOCAL Variable : public cuda::hpe::VariableDescriptor {
	CUdeviceptr ptr_;
    size_t size_;
public:
	Variable(const VariableDescriptor & v, CUmodule mod);
    size_t size() const;
    CUdeviceptr devPtr() const;
};

class GMAC_LOCAL Texture : public cuda::hpe::TextureDescriptor {
protected:
    CUtexref texRef_;

public:
	Texture(const TextureDescriptor & t, CUmodule mod);

    CUtexref texRef() const;
};

class Module;
typedef Module *ModulePtr;
typedef std::vector<ModulePtr> ModuleVector;

class GMAC_LOCAL ModuleDescriptor {
	friend class Module;

protected:
    typedef std::vector<ModuleDescriptor *> ModuleDescriptorVector;
    static ModuleDescriptorVector Modules_;
	const void *fatBin_;

    typedef std::vector<core::hpe::KernelDescriptor> KernelVector;
    typedef std::vector<VariableDescriptor>     VariableVector;
	typedef std::vector<TextureDescriptor>      TextureVector;

    KernelVector   kernels_;
	VariableVector variables_;
	VariableVector constants_;
	TextureVector  textures_;

public:
    ModuleDescriptor(const void * fatBin);

    void add(core::hpe::KernelDescriptor & k);
    void add(VariableDescriptor     & v);
    void add(TextureDescriptor      & t);

    static ModuleVector createModules();
};

typedef std::vector<ModuleDescriptor *> ModuleDescriptorVector;

class GMAC_LOCAL Module {
protected:

	std::vector<CUmodule> mods_;
	const void *fatBin_;

	typedef std::map<gmacVariable_t, Variable> VariableMap;
	typedef std::map<std::string, Variable> VariableNameMap;
	typedef std::map<gmacTexture_t, Texture> TextureMap;
    typedef std::map<gmac_kernel_id_t, Kernel *> KernelMap;

    VariableMap variables_;
	VariableMap constants_;
    VariableNameMap variablesByName_;
	VariableNameMap constantsByName_;
	TextureMap textures_;
    KernelMap kernels_;

public:
	Module(const ModuleDescriptorVector & dVector);
	~Module();

    void registerKernels(Mode &mode) const;

    const Variable *variable(gmacVariable_t key) const;
	const Variable *constant(gmacVariable_t key) const;
    const Variable *variableByName(std::string name) const;
	const Variable *constantByName(std::string name) const;
    const Texture  *texture(gmacTexture_t key) const;
};

}}}

#include "Module-impl.h"

#endif
