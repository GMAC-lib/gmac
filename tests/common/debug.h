#ifndef GMAC_TESTS_COMMON_DEBUG_H_
#define GMAC_TESTS_COMMON_DEBUG_H_

#if 0
#include <stdio.h>
#include <string.h>
#include <errno.h>

#ifdef CUDA_ERRORS
#include <cuda.h>
#define gmacGetErrorString cudaGetErrorString
#define gmacGetLastError cudaGetLastError
#else
#include "gmac.h"
#endif

#endif

#define FATAL()	\
	do {	\
		fprintf(stderr,"[%s:%d] : %s\n", __FILE__, __LINE__, \
			strerror(errno));	\
		exit(-1);	\
	} while(0)

#define CUFATAL()	\
	do {	\
		fprintf(stderr,"[%s:%d] : %s\n", __FILE__, __LINE__, \
			gmacGetErrorString(gmacGetLastError()));	\
		exit(-1);	\
	} while(0)

#endif
