#include <stdio.h>
#include <cstring>
#include <gmac/opencl.h>

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
	unsigned *ptr;

    assert(eclCompileSource(kernel) == eclSuccess);

	assert(eclMalloc((void **)&ptr, size * sizeof(unsigned)) == eclSuccess);

	// Call the kernel
    size_t globalSize = size;
    unsigned val = 1;

    eclMemset(ptr, 0, size * sizeof(unsigned));

    ecl_kernel kernel;

    assert(eclGetKernel("reset", &kernel) == eclSuccess);

    assert(eclSetKernelArgPtr(kernel, 0, ptr) == eclSuccess);
    assert(eclSetKernelArg(kernel, 1, sizeof(size), &size) == eclSuccess);
    assert(eclSetKernelArg(kernel, 2, sizeof(val), &val) == eclSuccess);
    assert(eclCallNDRange(kernel, 1, NULL, &globalSize, NULL) == eclSuccess);

	fprintf(stderr,"%d\n", check(ptr, size));

	fprintf(stderr, "Test partial memset: ");
	eclMemset(&ptr[size / 8], 0, 3 * size / 4 * sizeof(unsigned));
	fprintf(stderr,"%d\n", check(ptr, size / 4));

	fprintf(stderr,"Test full memset: ");
    memset(ptr, 0, size * sizeof(unsigned));

    assert(eclCallNDRange(kernel, 1, NULL, &globalSize, NULL) == eclSuccess);

	fprintf(stderr,"%d\n", check(ptr, size));

	fprintf(stderr, "Test partial memset: ");
	memset(&ptr[size / 8], 0, 3 * size / 4 * sizeof(unsigned));
	fprintf(stderr,"%d\n", check(ptr, size / 4));

    eclReleaseKernel(kernel);

	eclFree(ptr);

    return 0;
}
