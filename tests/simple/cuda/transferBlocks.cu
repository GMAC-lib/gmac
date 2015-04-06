#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <sys/time.h>

#include <map>
#include <algorithm>
#include <numeric>

#include <sys/mman.h>

#include <signal.h>

#include <pthread.h>

#include <cuda.h>
#define CUDA_ERRORS
#include "debug.h"
#include "utils.h"

#define MAXDOUBLE DBL_MAX
#define MINDOUBLE DBL_MIN

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define BANDWIDTH(s, t) ((s) * 8.0 / 1000.0 / (t))

enum TransferType {
    TRANSFER_IN  = 0,
    TRANSFER_OUT = 1
};

std::map<void *, int> blocks;

const char *pageLockedStr = "GMAC_PAGE_LOCKED";
const bool pageLockedDefault = false;
bool pageLocked = pageLockedDefault;

const char *minSizeStr = "GMAC_MIN";
const size_t minSizeDefault = 4 * 1024;
size_t minSize = minSizeDefault;

const char *maxSizeStr = "GMAC_MAX";
const size_t maxSizeDefault = 64 * 1024 * 1024;
size_t maxSize = maxSizeDefault;

const char *transferSizeStr = "GMAC_TRANSFER";
const size_t transferSizeDefault = 4 * maxSizeDefault;
size_t transferSize = transferSizeDefault;

const char *typeStr = "GMAC_TYPE";
const int typeDefault = TRANSFER_IN;
int type = TRANSFER_IN;


typedef unsigned long long usec_t;
typedef struct {
	double _time, _max, _min,
           _memcpy, _memcpy_max, _memcpy_min,
           _search, _search_max, _search_min,
           _sigsegv, _sigsegv_max, _sigsegv_min;
} stamp_t;

const int iters = 10;
//const size_t block_elems = 512;

static uint8_t *dev;
static uint8_t *cache;
static uint8_t *cpu;
static uint8_t *cpuTmp;

static uint8_t trash;
static size_t faults;

static struct sigaction defaultAction;

cudaStream_t stream;

#if defined(LINUX)
int signum = SIGSEGV;
#elif defined(DARWIN)
int signum = SIGBUS;
#endif

static size_t current_block_size;

void protect_block(void * block, size_t size, int prot)
{
    //fprintf(stdout, "Protecting %p: %zd %d\n", size, prot);
    assert(mprotect(block, size, prot) == 0);
}

void protect_buffer(void * buffer, size_t block_size, size_t buffer_size, int prot)
{
    //fprintf(stdout, "Protecting %p: %zd\n", buffer, buffer_size);
    for (off_t off = 0; off < buffer_size; off += block_size) {
        //fprintf(stdout, "Protecting %p: %zd %d\n", ((uint8_t *) buffer) + off, block_size, prot);
        protect_block(((uint8_t *) buffer) + off, block_size, prot);
    }
}

void segvHandler(int s, siginfo_t *info, void *ctx)
{   
#if 0
    mcontext_t *mCtx = &((ucontext_t *)ctx)->uc_mcontext;

#if defined(LINUX)
    unsigned long writeAccess = mCtx->gregs[REG_ERR] & 0x2;
#elif defined(DARWIN)
    unsigned long writeAccess = (*mCtx)->__es.__err & 0x2;
#endif
#endif
    void * addr = info->si_addr;
    protect_block(addr, current_block_size, PROT_READ | PROT_WRITE);
    faults++;

    //fprintf(stdout, "SIGSEGV done\n");
}

void program_sigsegv()
{
    struct sigaction segvAction;
    memset(&segvAction, 0, sizeof(segvAction));
    segvAction.sa_sigaction = segvHandler;
    segvAction.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&segvAction.sa_mask);

    if(sigaction(signum, &segvAction, &defaultAction) < 0) {
        fprintf(stderr, "sigaction: %s", strerror(errno));
        exit(-1);
    }
}

void restore_sigsegv()
{
    if(sigaction(signum, &defaultAction, NULL) < 0)
        fprintf(stderr, "sigaction: %s", strerror(errno));
}


inline usec_t get_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	usec_t tm = tv.tv_usec + 1000000 * tv.tv_sec;
	return tm;
}

struct param_t {
    TransferType type;
    stamp_t * stamp;
    size_t size;
    size_t totalSize;
};

void fill_cache()
{
    for (int i = 0; i < maxSize; i++) {
        cache[i] = 0;
    }
}

void fill_map(void * buffer, size_t block_size, size_t buffer_size)
{
    for (off_t off = 0; off < buffer_size; off += block_size) {
        blocks[((uint8_t *) buffer) + off] = 1;
    }
}

void reset_stamp(stamp_t * stamp)
{
    stamp->_time = 0.0;
    stamp->_max = MINDOUBLE;
    stamp->_min = MAXDOUBLE;
    stamp->_memcpy = 0.0;
    stamp->_memcpy_max = MINDOUBLE;
    stamp->_memcpy_min = MAXDOUBLE;
    stamp->_sigsegv = 0.0;
    stamp->_sigsegv_max = MINDOUBLE;
    stamp->_sigsegv_min = MAXDOUBLE;
    stamp->_search = 0.0;
    stamp->_search_max = MINDOUBLE;
    stamp->_search_min = MAXDOUBLE;
}

void * transfer(param_t param)
{
    stamp_t * stamp = param.stamp;
    reset_stamp(stamp);
    faults = 0;

    static double * time    = NULL;
    static double * memcopy = NULL;
    static double * sigsegv = NULL;
    static double * search  = NULL;

    if (time == NULL) time = new double[1024*1024];
    if (memcopy == NULL) memcopy = new double[1024*1024];
    if (sigsegv == NULL) sigsegv = new double[1024*1024];
    if (search == NULL) search = new double[1024*1024];

    for(int j = 0; j < iters; j++) {
        usec_t start, end;
        usec_t memcpy_start, memcpy_end;
        usec_t search_start, search_end;
        usec_t sigsegv_start, sigsegv_end;

        time[j]    = 0.0;
        memcopy[j] = 0.0;
        sigsegv[j] = 0.0;
        search[j]  = 0.0;

        current_block_size = param.size;
        if (param.type == TRANSFER_IN) {
            protect_buffer(cpu, param.size, param.totalSize, PROT_READ);
        } else {
            protect_buffer(cpu, param.size, param.totalSize, PROT_WRITE);
        }

        blocks.clear();
        fill_map(cpu, param.size, param.totalSize);
        fill_cache();
        if (param.type == TRANSFER_IN) {
            for (int c = 0; c < param.totalSize/param.size; c++) {
                search_start = get_time();
                blocks.find(cpu + param.size * c);
                search_end = get_time();

                if (pageLocked) {
                    // TRANSFER
                    start = get_time();
                    if(cudaMemcpyAsync(cpuTmp, dev + param.size * c, param.size, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
                        CUFATAL();
                    cudaStreamSynchronize(stream);
                    end = get_time();
                    // SIGSEGV
                    sigsegv_start = get_time();
                    (cpu + param.size * c)[0] = 0; 
                    sigsegv_end = get_time();
                    // MEMCPY
                    memcpy_start = get_time();
                    memcpy(cpu + param.size * c, cpuTmp, param.size);
                    memcpy_end = get_time();
                    memcopy[j] += memcpy_end - memcpy_start;
                } else {
                    // SIGSEGV
                    sigsegv_start = get_time();
                    (cpu + param.size * c)[0] = 0; 
                    sigsegv_end = get_time();
                    // TRANSFER TIME
                    start = get_time();
                    if(cudaMemcpy(cpu + param.size * c, dev + param.size * c, param.size, cudaMemcpyDeviceToHost) != cudaSuccess)
                        CUFATAL();
                    cudaThreadSynchronize();
                    end = get_time();
                }

                time[j] += end - start;
                search[j] += search_end - search_start;
                sigsegv[j] += sigsegv_end - sigsegv_start;
            }
        } else {
            for (int c = 0; c < param.totalSize/param.size; c++) {
                search_start = get_time();
                blocks.find(cpu + param.size * c);
                search_end = get_time();

                if (pageLocked) {
                    // SIGSEGV
                    sigsegv_start = get_time();
                    trash += (cpu + param.size * c)[0]; 
                    sigsegv_end = get_time();
                    // MEMCPY
                    memcpy_start = get_time();
                    memcpy(cpuTmp, cpu + param.size * c, param.size);
                    memcpy_end = get_time();
                    // TRANSFER
                    start = get_time();
                    if(cudaMemcpyAsync(dev + param.size * c, cpu + param.size * c, param.size, cudaMemcpyHostToDevice, stream) != cudaSuccess)
                        CUFATAL();
                    cudaStreamSynchronize(stream);
                    end = get_time();
                    memcopy[j] += memcpy_end - memcpy_start;
                } else {
                    // SIGSEGV
                    sigsegv_start = get_time();
                    trash += (cpu + param.size * c)[0]; 
                    sigsegv_end = get_time();
                    // TRANSFER
                    start = get_time();
                    if(cudaMemcpy(dev + param.size * c, cpu + param.size * c, param.size, cudaMemcpyHostToDevice) != cudaSuccess)
                        CUFATAL();
                    cudaThreadSynchronize();
                    end = get_time();
                }

                time[j] += end - start;
                search[j] += search_end - search_start;
                sigsegv[j] += sigsegv_end - sigsegv_start;
            }
        }
    }
    stamp->_time = std::accumulate(time, time + iters, 0.0)/iters;
    stamp->_max = *std::max_element(time, time + iters);
    stamp->_min = *std::min_element(time, time + iters);
    stamp->_memcpy = std::accumulate(memcopy, memcopy + iters, 0.0)/iters;
    stamp->_memcpy_max = *std::max_element(memcopy, memcopy + iters);
    stamp->_memcpy_min = *std::min_element(memcopy, memcopy + iters);
    stamp->_search = std::accumulate(search, search + iters, 0.0)/iters;
    stamp->_search_max = *std::max_element(search, search + iters);
    stamp->_search_min = *std::min_element(search, search + iters);
    stamp->_sigsegv = std::accumulate(sigsegv, sigsegv + iters, 0.0)/iters;
    stamp->_sigsegv_max = *std::max_element(sigsegv, sigsegv + iters);
    stamp->_sigsegv_min = *std::min_element(sigsegv, sigsegv + iters);

    for(int j = 0; j < iters; j++) {
        if (fabs(time[j] - stamp->_time) / stamp->_time > 0.1) printf("Fiesta en cabina1\n");
        if (fabs(memcopy[j] - stamp->_memcpy) / stamp->_memcpy > 0.1) printf("Fiesta en cabina2\n");
        if (fabs(search[j] - stamp->_search) / stamp->_search > 0.1) printf("Fiesta en cabina3\n");
        if (fabs(sigsegv[j] - stamp->_sigsegv) / stamp->_sigsegv > 0.1) printf("Fiesta en cabina4\n");
    }

    return NULL;
}


int main(int argc, char *argv[])
{
    setParam<bool>(&pageLocked, pageLockedStr, pageLockedDefault);
    setParam<size_t>(&maxSize, maxSizeStr, maxSizeDefault);
    setParam<size_t>(&minSize, minSizeStr, minSizeDefault);
    setParam<size_t>(&transferSize, transferSizeStr, transferSizeDefault);
    setParam<int>(&type, typeStr, typeDefault);

    program_sigsegv();

	stamp_t stamp;
    param_t param;
    param.stamp = &stamp;
    param.totalSize = transferSize;

    // Alloc & init input data
    if(pageLocked) {
        if(cudaStreamCreate(&stream) != cudaSuccess)
            FATAL();
        if(cudaHostAlloc((void **) &cpuTmp, maxSize, cudaHostAllocPortable) != cudaSuccess)
            FATAL();
    }
    if(posix_memalign((void **) &cpu, 4096, param.totalSize) != 0)
        FATAL();

	if (cudaMalloc((void **) &dev, param.totalSize) != cudaSuccess)
		CUFATAL();

    if((cache = (uint8_t *) malloc(maxSize)) == NULL)
		CUFATAL();

	// Transfer data
    if (pageLocked) {
        fprintf(stdout, "block_size\tn_blocks\ttransfer\tmemcpy\tsearch\tsigsegv");
    } else {
        fprintf(stdout, "block_size\tn_blocks\ttransfer\tsearch\tsigsegv");
    }
    /*
	fprintf(stdout, "Min In Bandwidth\tMin Out Bandwidth\t");
	fprintf(stdout, "Max In Bandwidth\tMax Out Bandwidth\n");
    */
	fprintf(stdout, "\n");
	for (int i = minSize; i <= maxSize; i *= 2) {
        param.size = i;
        param.type = (TransferType) type;
        transfer(param);
        #if 0
		if (i > 1024 * 1024) fprintf(stdout, "%dMB\t", i / 1024 / 1024);
		else if (i > 1024) fprintf(stdout, "%dKB\t", i / 1024);
        #endif
		fprintf(stdout, "%d\t%d\t", i, maxSize/i);

        if (pageLocked) {
            fprintf(stdout, "%f\t%f\t%f\t%f",
                                        stamp._time,
                                        stamp._memcpy,
                                        stamp._search,
                                        stamp._sigsegv);
        } else {
            fprintf(stdout, "%f\t%f\t%f",
                                        stamp._time,
                                        stamp._search,
                                        stamp._sigsegv);
        }
        
        fprintf(stdout, "\n");

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
	}

    if(pageLocked) {
        cudaFreeHost(cpu);
    } else {
        free(cpu);
    }

	// Release memory
	cudaFree(dev);

    restore_sigsegv();
}
