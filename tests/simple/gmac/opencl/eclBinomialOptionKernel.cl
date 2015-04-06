
static const char * code = " \
#define RISKFREE 0.02f \n \
#define VOLATILITY 0.30f \n \
\n \
__kernel \
void \
binomial_options( \
    int numSteps, \
    const __global float4* randArray, \
    __global float4* output, \
    __local float4* callA, \
    __local float4* callB) \n \
{\n \
    // load shared mem \n \
    unsigned int tid = get_local_id(0);\n \
    unsigned int bid = get_group_id(0);\n \
    \n \
    float4 inRand = randArray[bid];\n \
    \n \
    float4 s = (1.0f - inRand) * 5.0f + inRand * 30.f;\n \
    float4 x = (1.0f - inRand) * 1.0f + inRand * 100.f;\n \
    float4 optionYears = (1.0f - inRand) * 0.25f + inRand * 10.f; \n \
    float4 dt = optionYears * (1.0f / (float)numSteps);\n \
    float4 vsdt = VOLATILITY * sqrt(dt);\n \
    float4 rdt = RISKFREE * dt;\n \
    float4 r = exp(rdt);\n \
    float4 rInv = 1.0f / r;\n \
    float4 u = exp(vsdt);\n \
    float4 d = 1.0f / u;\n \
    float4 pu = (r - d)/(u - d);\n \
    float4 pd = 1.0f - pu;\n \
    float4 puByr = pu * rInv;\n \
    float4 pdByr = pd * rInv;\n \
    \n \
    float4 profit = s * exp(vsdt * (2.0f * tid - (float)numSteps)) - x;\n \
    callA[tid].x = profit.x > 0 ? profit.x : 0.0f;\n \
    callA[tid].y = profit.y > 0 ? profit.y : 0.0f;\n \
    callA[tid].z = profit.z > 0 ? profit.z: 0.0f;\n \
    callA[tid].w = profit.w > 0 ? profit.w: 0.0f;\n \
    \n \
    barrier(CLK_LOCAL_MEM_FENCE);\n \
    \n \
    for(int j = numSteps; j > 0; j -= 2)\n \
    {\n \
        if(tid < j)\n \
        {\n \
            callB[tid] = puByr * callA[tid] + pdByr * callA[tid + 1];\n \
        }\n \
        barrier(CLK_LOCAL_MEM_FENCE);\n \
        \n \
        if(tid < j - 1)\n \
        {\n \
            callA[tid] = puByr * callB[tid] + pdByr * callB[tid + 1];\n \
        }\n \
        barrier(CLK_LOCAL_MEM_FENCE);\n \
    }\n \
    \n \
    // write result for this block to global mem\n \
    if(tid == 0) output[bid] = callA[0];\n \
}\n \
";
