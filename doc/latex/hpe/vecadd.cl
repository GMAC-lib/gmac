__kernel void vecadd(__global float *c,
                     __global float const * restrict a,
                     __global float const * restrict b)
{
    c[get_global_id(0)] = a[get_global_id(0)]
                        + b[get_global_id(0)];
}
