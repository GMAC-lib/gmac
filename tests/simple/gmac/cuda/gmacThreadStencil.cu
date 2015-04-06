#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "barrier.h"

#include "utils.h"
#include "debug.h"

#include "gmacStencilCommon.cu"

const char * nIterStr = "GMAC_NITER";
const unsigned nIterDefault        = 4;

static unsigned nIter = 0;


int main(int argc, char *argv[])
{
	setParam<unsigned>(&dimRealElems, dimRealElemsStr, dimRealElemsDefault);
	setParam<unsigned>(&nIter, nIterStr, nIterDefault);

    assert(nIter > 0);
    assert(dimRealElems % 32 == 0);

    dimElems = dimRealElems + 2 * STENCIL;

    JobDescriptor * descriptors = new JobDescriptor[nIter];
    thread_t * nThread = new thread_t[nIter];

    if (nIter > 1) {
        barrier_init(&barrier, nIter);
    }

    for(unsigned n = 0; n < nIter; n++) {
        descriptors[n] = JobDescriptor();
        descriptors[n].gpus  = nIter;
        descriptors[n].gpuId = n;

        if (n > 0) {
            descriptors[n].prev = &descriptors[n - 1];
        } else {
            descriptors[n].prev = NULL;
        }

        if (n < nIter - 1) {
            descriptors[n].next = &descriptors[n + 1];
        } else {
            descriptors[n].next = NULL;
        }

        descriptors[n].gpuId = n;
        descriptors[n].dimRealElems = dimRealElems;
        descriptors[n].dimElems     = dimElems;
        descriptors[n].slices       = dimElems / nIter;
	}

	gmactime_t s, t;

	getTime(&s);


	for(unsigned n = 0; n < nIter; n++) {
		nThread[n] = thread_create(do_stencil, (void *) &descriptors[n]);
    }

	for(unsigned n = 0; n < nIter; n++) {
		thread_wait(nThread[n]);
	}

    if (nIter > 1) {
        barrier_destroy(&barrier);
    }

    getTime(&t);
	printTime(&s, &t, "Total: ", "\n");

    delete descriptors;
    delete nThread;

    return 0;
}
