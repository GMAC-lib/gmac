static const char * code = " \
\n \
#define S_LOWER_LIMIT 10.0f\n \
#define S_UPPER_LIMIT 100.0f\n \
#define K_LOWER_LIMIT 10.0f\n \
#define K_UPPER_LIMIT 100.0f\n \
#define T_LOWER_LIMIT 1.0f\n \
#define T_UPPER_LIMIT 10.0f\n \
#define R_LOWER_LIMIT 0.01f\n \
#define R_UPPER_LIMIT 0.05f\n \
#define SIGMA_LOWER_LIMIT 0.01f\n \
#define SIGMA_UPPER_LIMIT 0.10f\n \
\n \
/**\n \
 * @brief   Abromowitz Stegun approxmimation for PHI (Cumulative Normal Distribution Function)\n \
 * @param   X input value\n \
 * @param   phi pointer to store calculated CND of X\n \
 */\n \
void phi(float4 X, float4* phi)\n \
{\n \
    float4 y;\n \
    float4 absX;\n \
    float4 t;\n \
    float4 result;\n \
    \n \
    const float4 c1 = (float4)0.319381530f;\n \
    const float4 c2 = (float4)-0.356563782f;\n \
    const float4 c3 = (float4)1.781477937f;\n \
    const float4 c4 = (float4)-1.821255978f;\n \
    const float4 c5 = (float4)1.330274429f;\n \
    \n \
    const float4 zero = (float4)0.0f;\n \
    const float4 one = (float4)1.0f;\n \
    const float4 two = (float4)2.0f;\n \
    const float4 temp4 = (float4)0.2316419f;\n \
    \n \
    const float4 oneBySqrt2pi = (float4)0.398942280f;\n \
    \n \
    absX = fabs(X);\n \
    t = one/(one + temp4 * absX);\n \
    \n \
    y = one - oneBySqrt2pi * exp(-X*X/two) * t \
        * (c1 + t \
              * (c2 + t \
                    * (c3 + t \
                          * (c4 + t * c5))));\n \
    \n \
    result = (X < zero)? (one - y) : y;\n \
    \n \
    *phi = result;\n \
}\n \
\n \
/*\n \
 * @brief   Calculates the call and put prices by using Black Scholes model\n \
 * @param   s       Array of random values of current option price\n \
 * @param   sigma   Array of random values sigma\n \
 * @param   k       Array of random values strike price\n \
 * @param   t       Array of random values of expiration time\n \
 * @param   r       Array of random values of risk free interest rate\n \
 * @param   width   Width of call price or put price array\n \
 * @param   call    Array of calculated call price values\n \
 * @param   put     Array of calculated put price values\n \
 */\n \
__kernel \
void \
blackScholes(const __global float4 *randArray, \
             int width, \
             __global float4 *call, \
             __global float4 *put)\n \
{\n \
    float4 d1, d2;\n \
    float4 phiD1, phiD2;\n \
    float4 sigmaSqrtT;\n \
    float4 KexpMinusRT;\n \
    \n \
    size_t xPos = get_global_id(0);\n \
    size_t yPos = get_global_id(1);\n \
    float4 two = (float4)2.0f;\n \
    float4 inRand = randArray[yPos * width + xPos];\n \
    float4 S = S_LOWER_LIMIT * inRand + S_UPPER_LIMIT * (1.0f - inRand);\n \
    float4 K = K_LOWER_LIMIT * inRand + K_UPPER_LIMIT * (1.0f - inRand);\n \
    float4 T = T_LOWER_LIMIT * inRand + T_UPPER_LIMIT * (1.0f - inRand);\n \
    float4 R = R_LOWER_LIMIT * inRand + R_UPPER_LIMIT * (1.0f - inRand);\n \
    float4 sigmaVal = SIGMA_LOWER_LIMIT * inRand + SIGMA_UPPER_LIMIT * (1.0f - inRand);\n \
    \n \
    \n \
    sigmaSqrtT = sigmaVal * sqrt(T);\n \
    \n \
    d1 = (log(S/K) + (R + sigmaVal * sigmaVal / two)* T)/ sigmaSqrtT;\n \
    d2 = d1 - sigmaSqrtT;\n \
    \n \
    KexpMinusRT = K * exp(-R * T);\n \
    phi(d1, &phiD1), phi(d2, &phiD2);\n \
    call[yPos * width + xPos] = S * phiD1 - KexpMinusRT * phiD2;\n \
    phi(-d1, &phiD1), phi(-d2, &phiD2);\n \
    put[yPos * width + xPos]  = KexpMinusRT * phiD2 - S * phiD1;\n \
}\n \
";

