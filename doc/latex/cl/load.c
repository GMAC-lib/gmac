#include <gmac/cl.h>

. . .

    cl_helper helper;
    size_t platforms;
    cl_int error_code;

    /* Initialize OpenCL */
    if(oclInitHelpers(&platforms) != CL_SUCCESS) return -1;
    if(platforms == 0) return -1;

    /* Get helper of the first available platform */
    helper = clGetHelpers()[0];

    /* Load and compile the OpenCL kernel for the platform */
    error_code = clHelperLoadProgramFromFile(helper, kernel_file);
    if(error_code != CL_SUCCESS) {
        clReleaseHelpers();
        return -1;
    }
