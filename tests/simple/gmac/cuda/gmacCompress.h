#ifndef __TESTS_GMACCOMPRESS_H_
#define __TESTS_GMACCOMPRESS_H_

__global__ void dct(float *, float *, size_t, size_t);
__global__ void idct(float *, float *, size_t, size_t);
__global__ void quant(float *, float *,size_t, size_t, float);

#endif
