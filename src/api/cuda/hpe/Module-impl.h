#ifndef GMAC_API_CUDA_HPE_MODULE_IMPL_H_
#define GMAC_API_CUDA_HPE_MODULE_IMPL_H_

namespace __impl { namespace cuda { namespace hpe {

inline bool
VariableDescriptor::constant() const
{
    return constant_;
}

inline size_t
Variable::size() const
{
    return size_;
}

inline CUdeviceptr
Variable::devPtr() const
{
    return ptr_;
}

inline CUtexref
Texture::texRef() const
{
    return texRef_;
}

inline
void
ModuleDescriptor::add(core::hpe::KernelDescriptor & k)
{
    kernels_.push_back(k);
}

inline
void
ModuleDescriptor::add(VariableDescriptor & v)
{
    if (v.constant()) {
        constants_.push_back(v);
    } else {
        variables_.push_back(v);
    }
}

inline
void
ModuleDescriptor::add(TextureDescriptor & t)
{
    textures_.push_back(t);
}

inline const Variable *
Module::constant(gmacVariable_t key) const
{
    VariableMap::const_iterator v;
    v = constants_.find(key);
    if(v == constants_.end()) return NULL;
    return &v->second;
}

inline const Variable *
Module::variable(gmacVariable_t key) const
{
    VariableMap::const_iterator v;
    v = variables_.find(key);
    if(v == variables_.end()) return NULL;
    return &v->second;
}

inline const Variable *
Module::constantByName(std::string name) const
{
    VariableNameMap::const_iterator v;
    v = constantsByName_.find(name);
    if(v == constantsByName_.end()) return NULL;
    return &v->second;
}

inline const Variable *
Module::variableByName(std::string name) const
{
    VariableNameMap::const_iterator v;
    v = variablesByName_.find(name);
    if(v == variablesByName_.end()) return NULL;
    return &v->second;
}

inline const Texture *
Module::texture(gmacTexture_t key) const
{
    TextureMap::const_iterator t;
    t = textures_.find(key);
    if(t == textures_.end()) return NULL;
    return &t->second;
}


}}}

#endif
