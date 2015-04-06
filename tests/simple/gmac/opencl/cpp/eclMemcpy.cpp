#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <gmac/opencl>

#include "utils.h"

typedef cl_uchar uint8_t;

enum MemcpyType {
	GMAC_TO_GMAC = 1,
	HOST_TO_GMAC = 2,
	GMAC_TO_HOST = 3,
};

int type;
int typeDefault = GMAC_TO_GMAC;
const char *typeStr = "GMAC_MEMCPY_TYPE";

bool memcpyFn;
bool memcpyFnDefault = false;
const char *memcpyFnStr = "GMAC_MEMCPY_GMAC";

const size_t minCount = 1024;
const size_t maxCount = 16 * 1024 * 1024;

const char *kernel = "\
					 __kernel void null()\
					 {\
					 return;\
					 }\
					 ";

void init(uint8_t *ptr, int s, uint8_t v)
{
	for(int i = 0; i < s; i++) {
		ptr[i] = v;
	}
}

int memcpyTest(MemcpyType type, bool callKernel, void *(*memcpy_fn)(void *, const void *, size_t n))
{
	int error = 0;

	ecl::config globalSize (1);
	ecl::config localSize (1);

	ecl::error ret;
	ecl::kernel kernel("null", ret);
	assert(ret == eclSuccess);

	uint8_t *baseSrc = NULL;
	uint8_t *eclSrc = NULL;
	uint8_t *eclDst = NULL;

	baseSrc = (uint8_t *)malloc(maxCount);
	init(baseSrc, int(maxCount), 0xca);
	for (size_t count = minCount; count <= maxCount; count *= 2) {
		fprintf(stderr, "ALLOC: "FMT_SIZE"\n", count);

		if (type == GMAC_TO_GMAC) {
			assert(ecl::malloc((void **)&eclSrc, count) == eclSuccess);
			assert(ecl::malloc((void **)&eclDst, count) == eclSuccess);
		} else if (type == HOST_TO_GMAC) {
			eclSrc = (uint8_t *)malloc(count);
			assert(ecl::malloc((void **)&eclDst, count) == eclSuccess);
		} else if (type == GMAC_TO_HOST) {
			assert(ecl::malloc((void **)&eclSrc, count) == eclSuccess);
			eclDst = (uint8_t *)malloc(count);
		}

		for (size_t stride = 0, i = 1; stride < count/3; stride = i, i =  i * 2 - (i == 1? 0: 1)) {
			for (size_t copyCount = 1; copyCount < count/3; copyCount *= 2) {
				init(eclSrc + stride, int(copyCount), 0xca);
				if (stride == 0) {
					init(eclDst + stride, int(copyCount) + 1, 0);
				} else {
					init(eclDst + stride - 1, int(copyCount) + 2, 0);
				}
				assert(stride + copyCount <= count);

				if (callKernel) {
					ret = kernel.callNDRange(globalSize, localSize);
					assert(ret == eclSuccess);
				}
				memcpy_fn(eclDst + stride, eclSrc + stride, copyCount);

				int ret = memcmp(eclDst + stride, baseSrc + stride, copyCount);
				if (stride == 0) {
					ret = ret && (eclDst[stride - 1] == 0 && eclDst[stride + copyCount] == 0);
				} else {
					ret = ret && (eclDst[stride - 1] == 0 && eclDst[stride + copyCount] == 0);
				}

				if (ret != 0) {
#if 0
					fprintf(stderr, "Error: eclToGmacTest size: %zd, stride: %zd, copy: %zd\n",
						count    ,
						stride   ,
						copyCount);
#endif
					abort();
					error = 1;
					goto exit_test;
				}
#if 0
				for (unsigned k = 0; k < count; k++) {
					int ret = baseDst[k] != eclDst[k];
					if (ret != 0) {
						fprintf(stderr, "Error: eclToGmacTest size: %zd, stride: %zd, copy: %zd. Pos %u\n", count    ,
							stride   ,
							copyCount, k);
						error = 1;
					}
				}
#endif
			}
		}

		if (type == GMAC_TO_GMAC) {
			assert(ecl::free(eclSrc) == eclSuccess);
			assert(ecl::free(eclDst) == eclSuccess);
		} else if (type == HOST_TO_GMAC) {
			free(eclSrc);
			assert(ecl::free(eclDst) == eclSuccess);
		} else if (type == GMAC_TO_HOST) {
			assert(ecl::free(eclSrc) == eclSuccess);
			free(eclDst);
		}
	}
	free(baseSrc);

	return error;

exit_test:
	if (type == GMAC_TO_GMAC) {
		assert(ecl::free(eclSrc) == eclSuccess);
		assert(ecl::free(eclDst) == eclSuccess);
	} else if (type == HOST_TO_GMAC) {
		free(eclSrc);
		assert(ecl::free(eclDst) == eclSuccess);
	} else if (type == GMAC_TO_HOST) {
		assert(ecl::free(eclSrc) == eclSuccess);
		free(eclDst);
	}

	free(baseSrc);

	return error;
}

static void *eclMemcpyWrapper(void *dst, const void *src, size_t size)
{ 
	return ecl::memcpy(dst, src, size);
}

int main(int argc, char *argv[])
{
	setParam<int>(&type, typeStr, typeDefault);
	setParam<bool>(&memcpyFn, memcpyFnStr, memcpyFnDefault);

	assert(ecl::compileSource(kernel) == eclSuccess);

	int ret;

	if (memcpyFn == true) {
		ret = memcpyTest(MemcpyType(type), false, eclMemcpyWrapper);
		if (ret == 0) ret = memcpyTest(MemcpyType(type), true, eclMemcpyWrapper);
	} else {
		ret = memcpyTest(MemcpyType(type), false, memcpy);
		if (ret == 0) ret = memcpyTest(MemcpyType(type), true, memcpy);
	}

	return ret;
}
