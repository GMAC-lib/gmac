#include <cstdio>
#include <cstring>
#include <gmac/opencl>

const unsigned size = 4 * 1024 * 1024;

const char *kernel = "\
					 __kernel void reset(__global unsigned *a, unsigned size, unsigned v)\
					 {\
					 unsigned i = get_global_id(0);\
					 if(i >= size) return;\
					 \
					 a[i] += v;\
					 }\
					 ";

int check(unsigned *ptr, unsigned s)
{
	unsigned a = 0;
	for(unsigned i = 0; i < size; i++)
		a += ptr[i];
	return a - s;
}

int main(int argc, char *argv[])
{
	ecl::error ret;
	unsigned *ptr;

	ret = ecl::compileSource(kernel);
	assert(ret == eclSuccess);

	ret = ecl::malloc((void **)&ptr, size * sizeof(unsigned));
	assert(ret == eclSuccess);

	// Call the kernel

	unsigned val = 1;

	ecl::memset(ptr, 0, size * sizeof(unsigned));
	ecl::config globalSize(size);
	ecl::config localSize(1);  
	ecl::config globalWorkOffset(0);
	ecl::kernel kernel("reset", ret);
	assert(ret == eclSuccess);
#ifndef __GXX_EXPERIMENTAL_CXX0X__
	ret = kernel.setArg(0, ptr);
	assert(ret == eclSuccess);
	ret = kernel.setArg(1, size);
	assert(ret == eclSuccess);
	ret = kernel.setArg(2, val);
	assert(ret == eclSuccess);

	ret = kernel.callNDRange(globalSize,localSize,globalWorkOffset);
	assert(ret == eclSuccess);
#else
	assert(kernel(ptr, size, val)(globalSize,localSize,globalWorkOffset) == eclSuccess);
#endif

	fprintf(stderr,"%d\n", check(ptr, size));

	fprintf(stderr, "Test partial memset: ");

	ecl::memset(&ptr[size / 8], 0, 3 * size / 4 * sizeof(unsigned));
	fprintf(stderr,"%d\n", check(ptr, size / 4));

	fprintf(stderr,"Test full memset: ");
	memset(ptr, 0, size * sizeof(unsigned));

	ret = kernel.callNDRange(globalSize,localSize,globalWorkOffset);
	assert(ret == eclSuccess);

	fprintf(stderr,"%d\n", check(ptr, size));

	fprintf(stderr, "Test partial memset: ");
	memset(&ptr[size / 8], 0, 3 * size / 4 * sizeof(unsigned));
	fprintf(stderr,"%d\n", check(ptr, size / 4));

	ecl::free(ptr);

	return 0;
}
