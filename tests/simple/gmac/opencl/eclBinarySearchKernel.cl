
static const char * code = " \
__kernel void \
binarySearch(__global uint4 * outputArray, \
             __const __global uint  * sortedArray, \
             const   unsigned int findMe,\
             const   unsigned int globalLowerBound, \
             const   unsigned int globalUpperBound, \
             const   unsigned int subdivSize)\n \
{\n \
    unsigned int tid = get_global_id(0);\n \
    unsigned int lowerBound = globalLowerBound + subdivSize * tid;\n \
    unsigned int upperBound = lowerBound + subdivSize - 1;\n \
    unsigned int lowerBoundElement = sortedArray[lowerBound];\n \
    unsigned int upperBoundElement = sortedArray[upperBound];\n \
    if( (lowerBoundElement > findMe) || (upperBoundElement < findMe))\n \
    {\n \
        return;\n \
    }\n \
    else \n \
    {\n \
        outputArray[0].x = lowerBound;\n \
        outputArray[0].y = upperBound;\n \
        outputArray[0].w = 1;\n \
    }\n \
}\n \
";

