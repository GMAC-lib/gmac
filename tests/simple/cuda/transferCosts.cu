#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <sys/time.h>

#include <map>
#include <numeric>

#include <sys/mman.h>

#include <cuda.h>
#define CUDA_ERRORS
#include "debug.h"
#include "utils.h"

#include "cycle.h"

#define MAXDOUBLE DBL_MAX
#define MINDOUBLE DBL_MIN

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define BANDWIDTH(s, t) ((s) * 8.0 / 1000.0 / (t))

enum TransferType {
    TRANSFER_IN  = 0,
    TRANSFER_OUT = 1
};

const char *pageLockedStr = "GMAC_PAGE_LOCKED";
const bool pageLockedDefault = false;
bool pageLocked = pageLockedDefault;

const char *minSizeStr = "GMAC_MIN";
const size_t minSizeDefault = 4 * 1024;
size_t minSize = minSizeDefault;

const char *maxSizeStr = "GMAC_MAX";
const size_t maxSizeDefault = 64 * 1024 * 1024;
size_t maxSize = maxSizeDefault;

const char *typeStr = "GMAC_TYPE";
const int typeDefault = TRANSFER_IN;
int type = TRANSFER_IN;


typedef unsigned long long ticks;
typedef struct {
	double _time, _max, _min,
           _memcpy, _memcpy_max, _memcpy_min;
} stamp_t;

static uint8_t *dev;
static uint8_t *cache1,*cache2;
static uint8_t *cpu;
static uint8_t *cpuTmp;

cudaStream_t stream;

struct param_t {
    TransferType type;
    stamp_t * stamp;
    size_t size;
    size_t numTransfers;
};

void fill_cache()
{
    memcpy(cache2, cache1, maxSize);
}

void reset_stamp(stamp_t * stamp)
{
    stamp->_time = 0.0;
    stamp->_max = MINDOUBLE;
    stamp->_min = MAXDOUBLE;
    stamp->_memcpy = 0.0;
    stamp->_memcpy_max = MINDOUBLE;
    stamp->_memcpy_min = MAXDOUBLE;
}

void * transfer(param_t param)
{
    stamp_t * stamp = param.stamp;
    reset_stamp(stamp);

    ticks start, end;
    ticks memcpy_start, memcpy_end;

    static ticks * times   = NULL;
    static ticks * memcpys = NULL;

    if (times == NULL) {
        times   = new ticks[param.numTransfers];
        memcpys = new ticks[param.numTransfers];
    }

    if (param.type == TRANSFER_IN) {
        for (int c = 0; c < param.numTransfers; c++) {
            fill_cache();
            if (pageLocked) {
                // TRANSFER
                start = getticks();
                if(cudaMemcpyAsync(cpuTmp, dev, param.size, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
                    CUFATAL();
                cudaStreamSynchronize(stream);
                end = getticks();
                // MEMCPY
                memcpy_start = getticks();
                memcpy(cpu, cpuTmp, param.size);
                memcpy_end = getticks();
            } else {
                // TRANSFER TIME
                start = getticks();
                if(cudaMemcpy(cpu, dev, param.size, cudaMemcpyDeviceToHost) != cudaSuccess)
                    CUFATAL();
                cudaThreadSynchronize();
                end = getticks();
            }

            times[c]   = end - start;
            memcpys[c] = memcpy_end - memcpy_start;
        }
    } else {
        for (int c = 0; c < param.numTransfers; c++) {
            fill_cache();
            if (pageLocked) {
                // MEMCPY
                memcpy_start = getticks();
                memcpy(cpuTmp, cpu, param.size);
                memcpy_end = getticks();
                // TRANSFER
                start = getticks();
                if(cudaMemcpyAsync(dev, cpuTmp, param.size, cudaMemcpyHostToDevice, stream) != cudaSuccess)
                    CUFATAL();
                cudaStreamSynchronize(stream);
                end = getticks();
            } else {
                // TRANSFER
                start = getticks();
                if(cudaMemcpy(dev, cpu, param.size, cudaMemcpyHostToDevice) != cudaSuccess)
                    CUFATAL();
                cudaThreadSynchronize();
                end = getticks();
            }

            times[c]   = end - start;
            memcpys[c] = memcpy_end - memcpy_start;
        }
    }

    for (int c = 0; c < param.numTransfers; c++) {
        if (pageLocked) {
            fprintf(stdout, "%d,%d,%d,%d\n", c + 1, param.size, times[c], memcpys[c]);
        } else {
            fprintf(stdout, "%d,%d,%d\n", c + 1, param.size, times[c]);
        }
    }

    stamp->_time = std::accumulate(times, &times[param.numTransfers], 0) / double(param.numTransfers);
    stamp->_memcpy = std::accumulate(memcpys, &memcpys[param.numTransfers], 0) / double(param.numTransfers);

    return NULL;
}


int main(int argc, char *argv[])
{
    setParam<bool>(&pageLocked, pageLockedStr, pageLockedDefault);
    setParam<size_t>(&maxSize, maxSizeStr, maxSizeDefault);
    setParam<size_t>(&minSize, minSizeStr, minSizeDefault);
    setParam<int>(&type, typeStr, typeDefault);

	stamp_t stamp;
    param_t param;
    param.stamp = &stamp;
    param.numTransfers = 250;

    // Alloc & init input data
    if(pageLocked) {
        if(cudaStreamCreate(&stream) != cudaSuccess)
            FATAL();
        if(cudaHostAlloc((void **) &cpuTmp, maxSize, cudaHostAllocPortable) != cudaSuccess)
            FATAL();
    }
    if(posix_memalign((void **) &cpu, 4096, maxSize) != 0)
        FATAL();

	if (cudaMalloc((void **) &dev, maxSize) != cudaSuccess)
		CUFATAL();

    if((cache1 = (uint8_t *) malloc(maxSize)) == NULL)
		CUFATAL();
    if((cache2 = (uint8_t *) malloc(maxSize)) == NULL)
		CUFATAL();

    memset(cpu, 0, maxSize);
    memset(cpuTmp, 0, maxSize);

	// Transfer data
    if (pageLocked) {
        fprintf(stdout, "test,block_size,transfer,memcpy\n");
    } else {
        fprintf(stdout, "test,block_size,transfer\n");
    }
	for (int i = minSize; i <= maxSize; i *= 2) {
        param.size = i;
        param.type = (TransferType) type;
        transfer(param);
        #if 0
		if (i > 1024 * 1024) fprintf(stdout, "%dMB\t", i / 1024 / 1024);
		else if (i > 1024) fprintf(stdout, "%dKB\t", i / 1024);
		fprintf(stdout, "%d\t%d\t", i, maxSize/i);

        if (pageLocked) {
            fprintf(stdout, "%f\t%f", stamp._time,
                                      stamp._memcpy);
        } else {
            fprintf(stdout, "%f", stamp._time);
        }
        
        fprintf(stdout, "\n");
        #endif

#if 0
        param.type = TRANSFER_OUT;
        transfer(param);
		if (i > 1024 * 1024) fprintf(stdout, "%dMB\t", i / 1024 / 1024);
		else if (i > 1024) fprintf(stdout, "%dKB\t", i / 1024);
		fprintf(stdout, "%f\t%f\t", stamp._time, BANDWIDTH(param.totalSize, stamp._time + stamp._memcpy));
        if (pageLocked) {
            fprintf(stdout, "%f\t%f\t", BANDWIDTH(param.totalSize, stamp._time),
                                        BANDWIDTH(param.totalSize, stamp._memcpy));
        }
        fprintf(stdout, "\n");
#endif
        fprintf(stderr, "%d", cache2[i - 1]);
	}

    free(cache1);
    free(cache2);

    if(pageLocked) {
        cudaFreeHost(cpu);
    } else {
        free(cpu);
    }

	// Release memory
	cudaFree(dev);
}
