const char *kernel_code = "                                                                    \n \
                                                                                               \n \
#define PI 3.14159265358979                                                                    \n \
                                                                                               \n \
#define blockSize 16                                                                           \n \
                                                                                               \n \
__kernel void dct(__global float *out, __global float *in, unsigned width, unsigned height)    \n \
{                                                                                              \n \
    __local float tile[blockSize][blockSize];                                                  \n \
                                                                                               \n \
	unsigned l = get_global_id(1);                                                             \n \
	unsigned k = get_global_id(0);                                                             \n \
                                                                                               \n \
	/* Pre-compute some values */                                                              \n \
	float alpha, beta;                                                                         \n \
	if(k == 0) alpha = sqrt(1.0 / width);                                                      \n \
	else alpha = sqrt(2.0 / width);                                                            \n \
	if(l == 0) beta = sqrt(1.0 / height);                                                      \n \
	else beta = sqrt(2.0 / height);                                                            \n \
                                                                                               \n \
	float a = (PI / width) * k;                                                                \n \
	float b = (PI / height) * l;                                                               \n \
                                                                                               \n \
	float o = 0;                                                                               \n \
	for(unsigned j = 0; j < height; j += get_local_size(1)) {                                  \n \
		for(unsigned i = 0; i < width; i+= get_local_size(0)) {                                \n \
			/* Calculate n and m values */                                                     \n \
			unsigned y = j + get_local_id(1);                                                  \n \
			unsigned x = i + get_local_id(0);                                                  \n \
                                                                                               \n \
			/* Prefetch data in shared memory */                                               \n \
			if(x < width && y < height)                                                        \n \
				tile[get_local_id(0)][get_local_id(1)] = in[y * width + x];                    \n \
			mem_fence(CLK_GLOBAL_MEM_FENCE);                                                   \n \
                                                                                               \n \
			/* Compute the partial DCT */                                                      \n \
			for(unsigned m = 0; m < get_local_size(1); m++) {                                  \n \
				for(unsigned n = 0; n < get_local_size(0); n++) {                              \n \
					o += tile[m][n] * cos(a * (n + i + 0.5)) * cos(b * (m + j + 0.5));         \n \
				}                                                                              \n \
			}                                                                                  \n \
			                                                                                   \n \
			/* Done computing the DCT for the sub-block */                                     \n \
		}                                                                                      \n \
	}                                                                                          \n \
                                                                                               \n \
	if(k < width && l < height) {                                                              \n \
		out[(l * width) + k] = alpha * beta * o;                                               \n \
	}                                                                                          \n \
}                                                                                              \n \
                                                                                               \n \
__kernel void idct(__global float *out, __global float *in, unsigned width, unsigned height)   \n \
{                                                                                              \n \
    __local float tile[blockSize][blockSize];                                                  \n \
                                                                                               \n \
	unsigned y = get_global_id(1);                                                             \n \
	unsigned x = get_global_id(0);                                                             \n \
                                                                                               \n \
	/* Pre-compute some values */                                                              \n \
	float alpha, beta;                                                                         \n \
                                                                                               \n \
	float a = (PI / width) * (x + 0.5);                                                        \n \
	float b = (PI / height) * (y + 0.5);                                                       \n \
                                                                                               \n \
	float o = 0;                                                                               \n \
				                                                                               \n \
	for(unsigned j = 0; j < height; j += get_local_size(1)) {                                  \n \
		for(unsigned i = 0; i < width; i+= get_local_size(0)) {                                \n \
			/* Calculate n and m values */                                                     \n \
			unsigned l = j + get_local_id(1);                                                  \n \
			unsigned k = i + get_local_id(0);                                                  \n \
                                                                                               \n \
			/* Prefetch data in shared memory */                                               \n \
			if(i + get_local_id(0) < width && j + get_local_id(1) < height)                    \n \
				tile[get_local_id(0)][get_local_id(1)] = in[l * width + k];                    \n \
			mem_fence(CLK_GLOBAL_MEM_FENCE);                                                   \n \
                                                                                               \n \
			/* Compute the partial IDCT */                                                     \n \
			for(unsigned m = 0; m < get_local_size(1); m++) {                                  \n \
				for(unsigned n = 0; n < get_local_size(0); n++) {                              \n \
					/* Pre-compute some values */                                              \n \
					if((n + i) == 0) alpha = sqrt(1.0 / width);                                \n \
					else alpha = sqrt(2.0 / width);                                            \n \
					if((m + j) == 0) beta = sqrt(1.0 / height);                                \n \
					else beta = sqrt(2.0 / height);                                            \n \
					o += alpha * beta * tile[m][n] * cos(a * (n + i)) * cos(b * (m + j));      \n \
				}                                                                              \n \
			}                                                                                  \n \
			                                                                                   \n \
			/* Done computing the DCT for the sub-block */                                     \n \
		}                                                                                      \n \
	}                                                                                          \n \
                                                                                               \n \
	if(x < width && y < height) {                                                              \n \
		out[(y * width) + x] = o;                                                              \n \
	}                                                                                          \n \
}                                                                                              \n \
                                                                                               \n \
__kernel void quant(__global float *out, __global float *in, unsigned width, unsigned height, float k)         \n \
{                                                                                              \n \
    __local float tile[blockSize][blockSize];                                                  \n \
                                                                                               \n \
	unsigned y = get_global_id(1);                                                             \n \
	unsigned x = get_global_id(0);                                                             \n \
                                                                                               \n \
	if(x < width && y < height) {                                                              \n \
		float f = fabs(in[y * width + x]);                                                     \n \
		if(f > k) out[(y * width) + x] = f;                                                    \n \
		else out[(y * width) + x] = 0.0;                                                       \n \
	}                                                                                          \n \
}";
