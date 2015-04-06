#include <gmac/opencl.h>

. . .

    eclError_t error_code;
    error_code = eclCompileSourceFile(kernel_file);
    if(error != eclSuccess) error(error_code);
