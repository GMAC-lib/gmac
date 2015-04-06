static const char * code = " \
typedef struct _MonteCalroAttrib \n \
 { \n \
     float4 strikePrice; \n \
     float4 c1; \n \
     float4 c2; \n \
     float4 c3; \n \
     float4 initPrice; \n \
     float4 sigma; \n \
     float4 timeStep; \n \
 }MonteCarloAttrib;\n \
 void \n \
lshift128(uint4 input, uint shift, uint4* output) \n \
{ \n \
    unsigned int invshift = 32u - shift; \n \
    \n \
    uint4 temp; \n \
    temp.x = input.x << shift; \n \
    temp.y = (input.y << shift) | (input.x >> invshift); \n \
    temp.z = (input.z << shift) | (input.y >> invshift); \n \
    temp.w = (input.w << shift) | (input.z >> invshift); \n \
    \n \
    *output = temp; \n \
}\n \
void \n \
rshift128(uint4 input, uint shift, uint4* output) \n \
{\n \
    unsigned int invshift = 32u - shift; \n \
    \n \
    uint4 temp; \n \
    \n \
    temp.w = input.w >> shift; \n \
    temp.z = (input.z >> shift) | (input.w << invshift); \n \
    temp.y = (input.y >> shift) | (input.z << invshift); \n \
    temp.x = (input.x >> shift) | (input.y << invshift); \n \
    \n \
    *output = temp; \n \
}\n \
void generateRand(uint4 seed, \
                  float4 *gaussianRand1, \
                  float4 *gaussianRand2, \
                  uint4 *nextRand) \n \
{ \n \
    uint mulFactor = 4; \n \
    uint4 temp[8]; \n \
    \n \
    uint4 state1 = seed;\n \
    uint4 state2 = (uint4)(0); \n \
    uint4 state3 = (uint4)(0); \n \
    uint4 state4 = (uint4)(0); \n \
    uint4 state5 = (uint4)(0); \n \
    \n \
    uint stateMask = 1812433253u; \n \
    uint thirty = 30u; \n \
    uint4 mask4 = (uint4)(stateMask); \n \
    uint4 thirty4 = (uint4)(thirty); \n \
    uint4 one4 = (uint4)(1u); \n \
    uint4 two4 = (uint4)(2u); \n \
    uint4 three4 = (uint4)(3u); \n \
    uint4 four4 = (uint4)(4u); \n \
    \n \
    uint4 r1 = (uint4)(0); \n \
    uint4 r2 = (uint4)(0); \n \
    \n \
    uint4 a = (uint4)(0); \n \
    uint4 b = (uint4)(0); \n \
    \n \
    uint4 e = (uint4)(0); \n \
    uint4 f = (uint4)(0); \n \
    \n \
    unsigned int thirteen  = 13u; \n \
    unsigned int fifteen = 15u; \n \
    unsigned int shift = 8u * 3u; \n \
    \n \
    unsigned int mask11 = 0xfdff37ffu;\n \
    unsigned int mask12 = 0xef7f3f7du; \n \
    unsigned int mask13 = 0xff777b7du; \n \
    unsigned int mask14 = 0x7ff7fb2fu; \n \
    \n \
    \n \
    const float one = 1.0f; \n \
    const float intMax = 4294967296.0f; \n \
    const float PI = 3.14159265358979f; \n \
    const float two = 2.0f; \n \
    \n \
    float4 r; \n \
    float4 phi; \n \
    \n \
    float4 temp1; \n \
    float4 temp2; \n \
    \n \
    //Initializing states. \n \
    state2 = mask4 * (state1 ^ (state1 >> thirty4)) + one4; \n \
    state3 = mask4 * (state2 ^ (state2 >> thirty4)) + two4; \n \
    state4 = mask4 * (state3 ^ (state3 >> thirty4)) + three4; \n \
    state5 = mask4 * (state4 ^ (state4 >> thirty4)) + four4; \n \
    \n \
    uint i = 0; \n \
    for(i = 0; i < mulFactor; ++i) \n \
    { \n \
        switch(i) \n \
        { \n \
            case 0: \n \
                r1 = state4; \n \
                r2 = state5; \n \
                a = state1;\n \
                b = state3;\n \
                break; \n \
            case 1: \n \
                r1 = r2; \n \
                r2 = temp[0]; \n \
                a = state2; \n \
                b = state4; \n \
                break; \n \
            case 2: \n \
                r1 = r2; \n \
                r2 = temp[1]; \n \
                a = state3; \n \
                b = state5; \n \
                break; \n \
            case 3: \n \
                r1 = r2; \n \
                r2 = temp[2]; \n \
                a = state4; \n \
                b = state1; \n \
                break; \n \
            default: \n \
                break;  \n \
                \n \
        }\n \
        \n \
        lshift128(a, shift, &e); \n \
        rshift128(r1, shift, &f); \n \
\n \
        temp[i].x = a.x ^ e.x ^ ((b.x >> thirteen) & mask11) ^ f.x ^ (r2.x << fifteen);\n \
        temp[i].y = a.y ^ e.y ^ ((b.y >> thirteen) & mask12) ^ f.y ^ (r2.y << fifteen);\n \
        temp[i].z = a.z ^ e.z ^ ((b.z >> thirteen) & mask13) ^ f.z ^ (r2.z << fifteen);\n \
        temp[i].w = a.w ^ e.w ^ ((b.w >> thirteen) & mask14) ^ f.w ^ (r2.w << fifteen);\n \
    }\n \
    \n \
    temp1 = convert_float4(temp[0]) * one / intMax; \n \
    temp2 = convert_float4(temp[1]) * one / intMax; \n \
        \n \
    // Applying Box Mullar Transformations.\n \
    r = sqrt((-two) * log(temp1)); \n \
    phi  = two * PI * temp2;\n \
    *gaussianRand1 = r * cos(phi); \n \
    *gaussianRand2 = r * sin(phi); \n \
    *nextRand = temp[2]; \n \
\n \
}\n \
void \n \
calOutputs(float4 strikePrice, \
                float4 meanDeriv1, \
                float4  meanDeriv2,  \
                float4 meanPrice1, \
                float4 meanPrice2, \
                float4 *pathDeriv1, \
                float4 *pathDeriv2, \
                float4 *priceVec1, \
                float4 *priceVec2) \n \
{ \n \
    float4 temp1 = (float4)0.0f; \n \
    float4 temp2 = (float4)0.0f; \n \
    float4 temp3 = (float4)0.0f; \n \
    float4 temp4 = (float4)0.0f; \n \
    \n \
    float4 tempDiff1 = meanPrice1 - strikePrice; \n \
    float4 tempDiff2 = meanPrice2 - strikePrice; \n \
    if(tempDiff1.x > 0.0f)\n \
    { \n \
        temp1.x = 1.0f; \n \
        temp3.x = tempDiff1.x; \n \
    }\n \
    if(tempDiff1.y > 0.0f) \n \
    { \n \
        temp1.y = 1.0f; \n \
        temp3.y = tempDiff1.y ; \n \
    }\n \
    if(tempDiff1.z > 0.0f) \n \
    { \n \
        temp1.z = 1.0f; \n \
        temp3.z = tempDiff1.z; \n \
    }\n \
    if(tempDiff1.w > 0.0f)\n \
    {\n \
        temp1.w = 1.0f; \n \
        temp3.w = tempDiff1.w; \n \
    }\n \
\n \
    if(tempDiff2.x > 0.0f) \n \
    { \n \
        temp2.x = 1.0f;\n \
        temp4.x = tempDiff2.x; \n \
    } \n \
    if(tempDiff2.y > 0.0f)\n \
    { \n \
        temp2.y = 1.0f; \n \
        temp4.y = tempDiff2.y; \n \
    } \n \
    if(tempDiff2.z > 0.0f) \n \
    { \n \
        temp2.z = 1.0f; \n \
        temp4.z = tempDiff2.z; \n \
    } \n \
    if(tempDiff2.w > 0.0f) \n \
    {\n \
        temp2.w = 1.0f; \n \
        temp4.w = tempDiff2.w; \n \
    }\n \
    \n \
    *pathDeriv1 = meanDeriv1 * temp1; \n \
    *pathDeriv2 = meanDeriv2 * temp2; \n \
    *priceVec1 = temp3; \n \
    *priceVec2 = temp4; \n \
    }\n \
__kernel \n \
void \n \
calPriceVega(MonteCarloAttrib attrib, \
            int noOfSum, \
            int width, \
            __global uint4 *randArray, \
            __global float4 *priceSamples, \
            __global float4 *pathDeriv) \n \
{\n \
float4 strikePrice = attrib.strikePrice; \n \
        float4 c1 = attrib.c1; \n \
        float4 c2 = attrib.c2; \n \
        float4 c3 = attrib.c3; \n \
        float4 initPrice = attrib.initPrice; \n \
        float4 sigma = attrib.sigma; \n \
        float4 timeStep = attrib.timeStep; \n \
        \n \
        size_t xPos = get_global_id(0); \n \
        size_t yPos = get_global_id(1); \n \
        \n  \
        float4 temp = (float4)0.0f;\n \
        \n \
        float4 price1 = (float4)0.0f;\n \
        float4 price2 = (float4)0.0f;\n \
        float4 pathDeriv1 = (float4)0.0f; \n \
        float4 pathDeriv2 = (float4)0.0f; \n \
        \n \
        float4 trajPrice1 = initPrice;\n \
        float4 trajPrice2 = initPrice;\n \
        \n \
        float4 sumPrice1 = initPrice;\n \
        float4 sumPrice2 = initPrice;\n \
        \n \
        float4 meanPrice1 = temp; \n \
        float4 meanPrice2 = temp; \n \
        \n \
        float4 sumDeriv1 = temp;\n \
        float4 sumDeriv2 = temp;\n \
        \n \
        float4 meanDeriv1 = temp; \n \
        float4 meanDeriv2 = temp;\n \
        \n \
        float4 finalRandf1 = temp;\n \
        float4 finalRandf2 = temp;\n \
        \n \
        uint4 nextRand = randArray[yPos * width + xPos]; \n \
        \n \
        //Run the Monte Carlo simulation a total of Num_Sum - 1 times \n \
        for(int i = 1; i < noOfSum; i++) \n \
        {\n \
            uint4 tempRand = nextRand; \n \
            generateRand(tempRand, &finalRandf1, &finalRandf2, &nextRand); \n \
            \n \
            //Calculate the trajectory price and sum price for all trajectories \n \
            trajPrice1 = trajPrice1 * exp(c1 + c2 * finalRandf1); \n \
            trajPrice2 = trajPrice2 * exp(c1 + c2 * finalRandf2); \n \
            \n \
            sumPrice1 = sumPrice1 + trajPrice1; \n \
            sumPrice2 = sumPrice2 + trajPrice2; \n \
            \n \
            temp = c3 * timeStep * i; \n \
            \n \
            // Calculate the derivative price for all trajectories \n \
            sumDeriv1 = sumDeriv1 + trajPrice1 \
                        * ((log(trajPrice1 / initPrice) - temp) / sigma);\n \
                        \n \
            sumDeriv2 = sumDeriv2 + trajPrice2 \
                        * ((log(trajPrice2 / initPrice) - temp) / sigma);\n \
                        \n \
        }\n \
    \n \
        //Calculate the average price and “average derivative?of each simulated path \n \
        meanPrice1 = sumPrice1 / noOfSum; \n \
        meanPrice2 = sumPrice2 / noOfSum; \n \
        meanDeriv1 = sumDeriv1 / noOfSum; \n \
        meanDeriv2 = sumDeriv2 / noOfSum; \n \
        \n \
        calOutputs(strikePrice, meanDeriv1, meanDeriv2, meanPrice1, \
                    meanPrice2, &pathDeriv1, &pathDeriv2, &price1, &price2); \n \
\n \
        priceSamples[(yPos * width + xPos) * 2] = price1; \n \
        priceSamples[(yPos * width + xPos) * 2 + 1] = price2; \n \
        pathDeriv[(yPos * width + xPos) * 2] = pathDeriv1; \n \
        pathDeriv[(yPos * width + xPos) * 2 + 1] = pathDeriv2; \n \
        \n \
        }\n \
";
