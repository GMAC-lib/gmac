#include <stdio.h>
#include <gmac/cuda.h>

#include "utils.h"

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

__global__ void null()
{
	return;
}

inline
void init(uint8_t *ptr, int s, uint8_t v)
{
#if 0
	for(int i = 0; i < s; i++) {
		ptr[i] = v;
	}
#endif
    ::memset(ptr, v, s);
}

int memcpyTest(MemcpyType type, bool callKernel, void *(*memcpy_fn)(void *, const void *, size_t n))
{
    int error = 0;

    uint8_t *baseSrc = NULL;
    uint8_t *gmacSrc = NULL;
    uint8_t *gmacDst = NULL;

    baseSrc = (uint8_t *)malloc(maxCount);
    init(baseSrc, int(maxCount), 0xca);
    for (size_t count = minCount; count <= maxCount; count *= 2) {
        fprintf(stderr, "ALLOC: "FMT_SIZE"\n", count);

        if (type == GMAC_TO_GMAC) {
            assert(gmacMalloc((void **)&gmacSrc, count) == gmacSuccess);
            assert(gmacMalloc((void **)&gmacDst, count) == gmacSuccess);
        } else if (type == HOST_TO_GMAC) {
            gmacSrc = (uint8_t *)malloc(count);
            assert(gmacMalloc((void **)&gmacDst, count) == gmacSuccess);
        } else if (type == GMAC_TO_HOST) {
            assert(gmacMalloc((void **)&gmacSrc, count) == gmacSuccess);
            gmacDst = (uint8_t *)malloc(count);
        }

        for (size_t stride = 0, i = 1; stride < count/3; stride = i, i =  i * 2 - (i == 1? 0: 1)) {
            for (size_t copyCount = 1; copyCount < count/3; copyCount *= 2) {
                init(gmacSrc + stride, int(copyCount), 0xca);
                if (stride == 0) {
                    init(gmacDst + stride, int(copyCount) + 1, 0);
                } else {
                    init(gmacDst + stride - 1, int(copyCount) + 2, 0);
                }
                assert(stride + copyCount <= count);

                if (callKernel) {
                    null<<<1, 1>>>();
                    assert(gmacThreadSynchronize() == gmacSuccess);
                }
                memcpy_fn(gmacDst + stride, gmacSrc + stride, copyCount);

                int ret = memcmp(gmacDst + stride, baseSrc + stride, copyCount);
                if (stride == 0) {
                    ret = ret && (gmacDst[stride + copyCount] == 0);
                } else {
                    ret = ret && (gmacDst[stride - 1] == 0 && gmacDst[stride + copyCount] == 0);
                }

                if (ret != 0) {
#if 0
                    fprintf(stderr, "Error: gmacToGmacTest size: %zd, stride: %zd, copy: %zd\n",
                            count    ,
                            stride   ,
                            copyCount);
#endif
                    error = 1;
                    goto exit_test;
                }
#if 0
                for (unsigned k = 0; k < count; k++) {
                    int ret = baseDst[k] != gmacDst[k];
                    if (ret != 0) {
                        fprintf(stderr, "Error: gmacToGmacTest size: %zd, stride: %zd, copy: %zd. Pos %u\n", count    ,
                                stride   ,
                                copyCount, k);
                        error = 1;
                    }
                }
#endif
            }
        }

        if (type == GMAC_TO_GMAC) {
            assert(gmacFree(gmacSrc) == gmacSuccess);
            assert(gmacFree(gmacDst) == gmacSuccess);
        } else if (type == HOST_TO_GMAC) {
            free(gmacSrc);
            assert(gmacFree(gmacDst) == gmacSuccess);
        } else if (type == GMAC_TO_HOST) {
            assert(gmacFree(gmacSrc) == gmacSuccess);
            free(gmacDst);
        }
    }
    free(baseSrc);

    return error;

exit_test:
    if (type == GMAC_TO_GMAC) {
        assert(gmacFree(gmacSrc) == gmacSuccess);
        assert(gmacFree(gmacDst) == gmacSuccess);
    } else if (type == HOST_TO_GMAC) {
        free(gmacSrc);
        assert(gmacFree(gmacDst) == gmacSuccess);
    } else if (type == GMAC_TO_HOST) {
        assert(gmacFree(gmacSrc) == gmacSuccess);
        free(gmacDst);
    }

    free(baseSrc);

    return error;
}

static void *gmacMemcpyWrapper(void *dst, const void *src, size_t size)
{
	return gmacMemcpy(dst, src, size);
}

int main(int argc, char *argv[])
{
	setParam<int>(&type, typeStr, typeDefault);
	setParam<bool>(&memcpyFn, memcpyFnStr, memcpyFnDefault);

    int ret = 0;
    
    if (memcpyFn == true) {
        fprintf(stderr, "Using GMAC memcpy\n");
        ret = memcpyTest(MemcpyType(type), false, gmacMemcpyWrapper);
        if (ret == 0) ret = memcpyTest(MemcpyType(type), true, gmacMemcpyWrapper);
    } else {
        fprintf(stderr, "Using stdc memcpy\n");
        ret = memcpyTest(MemcpyType(type), false, memcpy);
        if (ret == 0) ret = memcpyTest(MemcpyType(type), true, memcpy);
    }

    return ret;
}
