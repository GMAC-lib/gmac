static const char * code = " \
__kernel \
void bitonicSort(__global uint * theArray, \
                 const uint stage, \
                 const uint passOfStage, \
                 const uint width, \
                 const uint direction) \n \
{\n \
    uint sortIncreasing = direction; \n \
    uint threadId = get_global_id(0); \n \
    \n \
    uint pairDistance = 1 << (stage - passOfStage);\n \
    uint blockWidth   = 2 * pairDistance;\n \
    \n \
    uint leftId = (threadId % pairDistance) \
                   + (threadId / pairDistance) * blockWidth;\n \
    \n \
    uint rightId = leftId + pairDistance;\n \
    \n \
    uint leftElement = theArray[leftId];\n \
    uint rightElement = theArray[rightId];\n \
    \n \
    uint sameDirectionBlockWidth = 1 << stage;\n \
    \n \
    if((threadId/sameDirectionBlockWidth) % 2 == 1)\n \
        sortIncreasing = 1 - sortIncreasing;\n \
    \n \
    uint greater;\n \
    uint lesser;\n \
    if(leftElement > rightElement)\n \
    {\n \
        greater = leftElement;\n \
        lesser  = rightElement;\n \
    }\n \
    else\n \
    {\n \
        greater = rightElement;\n \
        lesser  = leftElement;\n \
    }\n \
    \n \
    if(sortIncreasing)\n \
    {\n \
        theArray[leftId]  = lesser;\n \
        theArray[rightId] = greater;\n \
    }\n \
    else\n \
    {\n \
        theArray[leftId]  = greater;\n \
        theArray[rightId] = lesser;\n \
    }\n \
}\n \
";
