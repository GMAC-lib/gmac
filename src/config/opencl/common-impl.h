#ifndef GMAC_CONFIG_OPENCL_COMMON_IMPL_H_
#define GMAC_CONFIG_OPENCL_COMMON_IMPL_H_

#include <cstdio>


template<typename T>
inline _opencl_ptr_t _opencl_ptr_t::operator+(const T &off) const
{
    cl_int error_code = CL_SUCCESS;
    size_t size = 0;


    /* Sanity check */
    if(base_ == 0) return _opencl_ptr_t(cl_mem(0));
    if(off == 0) return _opencl_ptr_t(base_);

    error_code = clGetMemObjectInfo(base_, CL_MEM_SIZE, sizeof(size), &size, NULL);

    /* Sanity Checks */
    if(error_code != CL_SUCCESS) return _opencl_ptr_t(cl_mem(0));
    if(size_t(off) >= size) return _opencl_ptr_t(cl_mem(0));


    cl_buffer_region region;
    region.origin = off;
    region.size = size - off;

    cl_mem sub = clCreateSubBuffer(base_, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &error_code);

    if(error_code != CL_SUCCESS) return _opencl_ptr_t(cl_mem(0));
    return _opencl_ptr_t(sub);
}

#endif
