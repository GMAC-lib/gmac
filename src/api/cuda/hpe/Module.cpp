#include "hpe/init.h"

#include "api/cuda/hpe/Module.h"
#include "api/cuda/hpe/Mode.h"

namespace __impl { namespace cuda { namespace hpe {

ModuleDescriptor::ModuleDescriptorVector ModuleDescriptor::Modules_;

VariableDescriptor::VariableDescriptor(const char *name, gmacVariable_t key, bool constant) :
    core::hpe::Descriptor<gmacVariable_t>(name, key),
    constant_(constant)
{
}

Variable::Variable(const VariableDescriptor & v, CUmodule mod) :
    VariableDescriptor(v.getName(), v.key(), v.constant())
{
#if CUDA_VERSION > 3010
    size_t tmp;
#else
    unsigned int tmp;
#endif
    TRACE(LOCAL, "Creating new accelerator variable: %s", v.getName());
    CUresult ret = cuModuleGetGlobal(&ptr_, &tmp, mod, getName());
    ASSERTION(ret == CUDA_SUCCESS);
    size_ = tmp;
}

Texture::Texture(const TextureDescriptor & t, CUmodule mod) :
    TextureDescriptor(t.getName(), t.key())
{
    CUresult ret = cuModuleGetTexRef(&texRef_, mod, getName());
    ASSERTION(ret == CUDA_SUCCESS);
}

ModuleDescriptor::ModuleDescriptor(const void *fatBin) :
    fatBin_(fatBin)
{
    TRACE(LOCAL, "Creating module descriptor: %p", fatBin_);
    ASSERTION(this != NULL);
    Modules_.push_back(this);
}

ModuleVector
ModuleDescriptor::createModules()
{
    TRACE(GLOBAL, "Creating modules");
    ModuleVector modules;

    ModulePtr ptr = new cuda::hpe::Module(Modules_);
    modules.push_back(ptr);
    return modules;
}

static const int CUDA_MAGIC = 0x466243b1;

struct GMAC_LOCAL FatBinDesc {
    int magic; int v; const unsigned long long* data; char* f;
};

Module::Module(const ModuleDescriptorVector & dVector)
{
    CUmodule mod;
    ModuleDescriptorVector::const_iterator it;
    for (it = dVector.begin(); it != dVector.end(); ++it) {
        ModuleDescriptor &d = *(*it);
        const void *fatBin_ = d.fatBin_;
        TRACE(LOCAL, "Module image: %p", fatBin_);
        CUresult res;

        FatBinDesc *desc = (FatBinDesc *)fatBin_;
        if (desc->magic == CUDA_MAGIC) {
            res = cuModuleLoadFatBinary(&mod, desc->data);
        } else {
            res = cuModuleLoadFatBinary(&mod, fatBin_);
        }
        CFATAL(res == CUDA_SUCCESS, "Error loading module: %d", res);

        ModuleDescriptor::KernelVector::const_iterator k;
        for (k = d.kernels_.begin(); k != d.kernels_.end(); ++k) {
            TRACE(LOCAL, "Registering kernel: %s", k->getName());
            kernels_.insert(KernelMap::value_type(k->key(), new cuda::hpe::Kernel(*k, mod)));
        }

        ModuleDescriptor::VariableVector::const_iterator v;
        for (v = d.variables_.begin(); v != d.variables_.end(); ++v) {
            VariableNameMap::const_iterator f;
            f = variablesByName_.find(v->getName());
            if (f != variablesByName_.end()) {
                FATAL("Variable already registered: %s", v->getName());
            } else {
                TRACE(LOCAL, "Registering variable: %s", v->getName());
                variables_.insert(VariableMap::value_type(v->key(), Variable(*v, mod)));
                variablesByName_.insert(VariableNameMap::value_type(std::string(v->getName()), Variable(*v, mod)));
            }
        }

        for (v = d.constants_.begin(); v != d.constants_.end(); ++v) {
            VariableNameMap::const_iterator f;
            f = constantsByName_.find(v->getName());
            if (f != constantsByName_.end()) {
                FATAL("Constant already registered: %s", v->getName());
            }
            TRACE(LOCAL, "Registering constant: %s", v->getName());
            constants_.insert(VariableMap::value_type(v->key(), Variable(*v, mod)));
            constantsByName_.insert(VariableNameMap::value_type(std::string(v->getName()), Variable(*v, mod)));
        }

        ModuleDescriptor::TextureVector::const_iterator t;
        for (t = d.textures_.begin(); t != d.textures_.end(); ++t) {
            textures_.insert(TextureMap::value_type(t->key(), Texture(*t, mod)));
        }

        mods_.push_back(mod);
    }
}

Module::~Module()
{
    KernelMap::iterator it;
    for (it = kernels_.begin(); it != kernels_.end(); ++it) {
        delete it->second;
    }
#ifdef CALL_CUDA_ON_DESTRUCTION
    std::vector<CUmodule>::const_iterator m;
    for(m = mods_.begin(); m != mods_.end(); ++m) {
        CUresult ret = cuModuleUnload(*m);
        ASSERTION(ret == CUDA_SUCCESS);
    }
    mods_.clear();
#endif
}

void Module::registerKernels(Mode &mode) const
{
    KernelMap::const_iterator k;
    for (k = kernels_.begin(); k != kernels_.end(); ++k) {
        mode.registerKernel(k->first, *k->second);
    }
}

}}}
