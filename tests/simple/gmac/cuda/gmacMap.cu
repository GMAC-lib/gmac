#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac.h>

#include "utils.h"
#include "debug.h"

#if !defined(_MSC_VER)
#include <sys/mman.h>
#include <map>

static std::map<void *, size_t> allocMap;
#endif


const char *vecSizeStr = "GMAC_VECSIZE";
const size_t vecSizeDefault = 4 * 1024 * 1024;

size_t vecSize = 0;
const size_t blockSize = 512;

const char *partsStr = "GMAC_PARTITIONS";
const unsigned partsDefault = 2;

unsigned parts = 0;

const char *msg = "Done!";

__global__ void vecAdd(float *c, const float *a, const float *b, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= size) return;

    c[i] = a[i] + b[i];
}

static void *aligned_malloc(size_t size)
{
#if defined(_MSC_VER)
	SYSTEM_INFO info;
	GetSystemInfo(&info);
	return _aligned_malloc(size, info.dwPageSize);
#else
	void *addr = mmap(NULL, size + sizeof(void *), PROT_READ | PROT_WRITE, MAP_ANONYMOUS, -1, 0);
    if(addr != NULL) allocMap[addr] = size;
    return addr;
#endif
}

static void aligned_free(void *addr)
{
#if defined(_MSC_VER)
	_aligned_free(addr);
#else
    std::map<void *, size_t>::iterator i;
    i = allocMap.find(addr);
    assert(i != allocMap.end());
	munmap(addr, i->second);
    allocMap.erase(i);
#endif
}

int main(int argc, char *argv[])
{
	float *a, *b, *c;
	gmactime_t s, t;

	setParam<size_t>(&vecSize, vecSizeStr, vecSizeDefault);
	setParam<unsigned>(&parts, partsStr,   partsDefault);
	fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);
	fprintf(stdout, "Partitions: %u\n", parts);

    size_t partSize = vecSize / parts;

    size_t vecBytes  = vecSize  * sizeof(float);
    size_t partBytes = partSize * sizeof(float);

    getTime(&s);
    // Alloc host data
    assert((a = (float *)aligned_malloc(vecBytes)) != NULL);
    assert((b = (float *)aligned_malloc(vecBytes)) != NULL);
    assert((c = (float *)aligned_malloc(vecBytes)) != NULL);
    getTime(&t);
    printTime(&s, &t, "Host alloc: ", "\n");

    // Init data
    getTime(&s);
    randInit(a, vecSize);
    randInit(b, vecSize);
    getTime(&t);
    printTime(&s, &t, "Init: ", "\n");

    float sum = 0.f;
    for(unsigned i = 0; i < vecSize; i++) {
        sum += a[i] + b[i];
    }

    for (unsigned i = 0; i < parts; i++) {
        float * partA = a + i * partSize;
        float * partB = b + i * partSize;
        float * partC = c + i * partSize;
        fprintf(stdout, "> Partition: %u\n", i);
        getTime(&s);
        assert(gmacMap(partA, partBytes, GMAC_PROT_READ) == gmacSuccess);
        assert(gmacMap(partB, partBytes, GMAC_PROT_READ) == gmacSuccess);
        assert(gmacMap(partC, partBytes, GMAC_PROT_WRITE) == gmacSuccess);
        getTime(&t);
        printTime(&s, &t, "  - Map: ", "\n");

        // Call the kernel
        getTime(&s);
        dim3 Db(blockSize);
        dim3 Dg((unsigned long)partSize / blockSize);
        if(partSize % blockSize) Dg.x++;
        vecAdd<<<Dg, Db>>>(gmacPtr(partC), gmacPtr(partA), gmacPtr(partB), partSize);
        if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
        getTime(&t);
        printTime(&s, &t, "  - Run: ", "\n");

        getTime(&s);
        assert(gmacUnmap(partA, partBytes) == gmacSuccess);
        assert(gmacUnmap(partB, partBytes) == gmacSuccess);
        assert(gmacUnmap(partC, partBytes) == gmacSuccess);
        getTime(&t);
        printTime(&s, &t, "  - Unmap: ", "\n");
    }

    getTime(&s);
    float error = 0.f;
    float check = 0.f;
    for(unsigned i = 0; i < vecSize; i++) {
        error += c[i] - (a[i] + b[i]);
        check += a[i] + b[i];
    }
    if (sum != check) {
        fprintf(stderr, "Checksum mismatch: %f vs %f\n", sum, check);
    }
    getTime(&t);
    printTime(&s, &t, "Check: ", "\n");

    fprintf(stderr, "Error: %f\n", error);

    aligned_free(a);
	aligned_free(b);
	aligned_free(c);

    return error != 0;
}
