/* ============================================================

Copyright (c) 2009 Advanced Micro Devices, Inc.  All rights reserved.

Redistribution and use of this material is permitted under the following
conditions:

Redistributions must retain the above copyright notice and all terms of this
license.

In no event shall anyone redistributing or accessing or using this material
commence or participate in any arbitration or legal action relating to this
material against Advanced Micro Devices, Inc. or any copyright holders or
contributors. The foregoing shall survive any expiration or termination of
this license or any agreement or access or use related to this material.

ANY BREACH OF ANY TERM OF THIS LICENSE SHALL RESULT IN THE IMMEDIATE REVOCATION
OF ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE THIS MATERIAL.

THIS MATERIAL IS PROVIDED BY ADVANCED MICRO DEVICES, INC. AND ANY COPYRIGHT
HOLDERS AND CONTRIBUTORS "AS IS" IN ITS CURRENT CONDITION AND WITHOUT ANY
REPRESENTATIONS, GUARANTEE, OR WARRANTY OF ANY KIND OR IN ANY WAY RELATED TO
SUPPORT, INDEMNITY, ERROR FREE OR UNINTERRUPTED OPERA TION, OR THAT IT IS FREE
FROM DEFECTS OR VIRUSES.  ALL OBLIGATIONS ARE HEREBY DISCLAIMED - WHETHER
EXPRESS, IMPLIED, OR STATUTORY - INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED
WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
ACCURACY, COMPLETENESS, OPERABILITY, QUALITY OF SERVICE, OR NON-INFRINGEMENT.
IN NO EVENT SHALL ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, PUNITIVE,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, REVENUE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED OR BASED ON ANY THEORY OF LIABILITY
ARISING IN ANY WAY RELATED TO THIS MATERIAL, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE. THE ENTIRE AND AGGREGATE LIABILITY OF ADVANCED MICRO DEVICES,
INC. AND ANY COPYRIGHT HOLDERS AND CONTRIBUTORS SHALL NOT EXCEED TEN DOLLARS
(US $10.00). ANYONE REDISTRIBUTING OR ACCESSING OR USING THIS MATERIAL ACCEPTS
THIS ALLOCATION OF RISK AND AGREES TO RELEASE ADVANCED MICRO DEVICES, INC. AND
ANY COPYRIGHT HOLDERS AND CONTRIBUTORS FROM ANY AND ALL LIABILITIES,
OBLIGATIONS, CLAIMS, OR DEMANDS IN EXCESS OF TEN DOLLARS (US $10.00). THE
FOREGOING ARE ESSENTIAL TERMS OF THIS LICENSE AND, IF ANY OF THESE TERMS ARE
CONSTRUED AS UNENFORCEABLE, FAIL IN ESSENTIAL PURPOSE, OR BECOME VOID OR
DETRIMENTAL TO ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR
CONTRIBUTORS FOR ANY REASON, THEN ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE
THIS MATERIAL SHALL TERMINATE IMMEDIATELY. MOREOVER, THE FOREGOING SHALL
SURVIVE ANY EXPIRATION OR TERMINATION OF THIS LICENSE OR ANY AGREEMENT OR
ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE IS HEREBY PROVIDED, AND BY REDISTRIBUTING OR ACCESSING OR USING THIS
MATERIAL SUCH NOTICE IS ACKNOWLEDGED, THAT THIS MATERIAL MAY BE SUBJECT TO
RESTRICTIONS UNDER THE LAWS AND REGULATIONS OF THE UNITED STATES OR OTHER
COUNTRIES, WHICH INCLUDE BUT ARE NOT LIMITED TO, U.S. EXPORT CONTROL LAWS SUCH
AS THE EXPORT ADMINISTRATION REGULATIONS AND NATIONAL SECURITY CONTROLS AS
DEFINED THEREUNDER, AS WELL AS STATE DEPARTMENT CONTROLS UNDER THE U.S.
MUNITIONS LIST. THIS MATERIAL MAY NOT BE USED, RELEASED, TRANSFERRED, IMPORTED,
EXPORTED AND/OR RE-EXPORTED IN ANY MANNER PROHIBITED UNDER ANY APPLICABLE LAWS,
INCLUDING U.S. EXPORT CONTROL LAWS REGARDING SPECIFICALLY DESIGNATED PERSONS,
COUNTRIES AND NATIONALS OF COUNTRIES SUBJECT TO NATIONAL SECURITY CONTROLS.
MOREOVER, THE FOREGOING SHALL SURVIVE ANY EXPIRATION OR TERMINATION OF ANY
LICENSE OR AGREEMENT OR ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE REGARDING THE U.S. GOVERNMENT AND DOD AGENCIES: This material is
provided with "RESTRICTED RIGHTS" and/or "LIMITED RIGHTS" as applicable to
computer software and technical data, respectively. Use, duplication,
distribution or disclosure by the U.S. Government and/or DOD agencies is
subject to the full extent of restrictions in all applicable regulations,
including those found at FAR52.227 and DFARS252.227 et seq. and any successor
regulations thereof. Use of this material by the U.S. Government and/or DOD
agencies is acknowledgment of the proprietary rights of any copyright holders
and contributors, including those of Advanced Micro Devices, Inc., as well as
the provisions of FAR52.227-14 through 23 regarding privately developed and/or
commercial computer software.

This license forms the entire agreement regarding the subject matter hereof and
supersedes all proposals and prior discussions and writings between the parties
with respect thereto. This license does not affect any ownership, rights, title,
or interest in, or relating to, this material. No terms of this license can be
modified or waived, and no breach of this license can be excused, unless done
so in a writing signed by all affected parties. Each term of this license is
separately enforceable. If any term of this license is determined to be or
becomes unenforceable or illegal, such term shall be reformed to the minimum
extent necessary in order for this license to remain in effect in accordance
with its terms as modified by such reformation. This license shall be governed
by and construed in accordance with the laws of the State of Texas without
regard to rules on conflicts of law of any state or jurisdiction or the United
Nations Convention on the International Sale of Goods. All disputes arising out
of this license shall be subject to the jurisdiction of the federal and state
courts in Austin, Texas, and all defenses are hereby waived concerning personal
jurisdiction and venue of these courts.

============================================================ */


#include<GL/glut.h>
#include<malloc.h>

#include <cmath>
#include <iostream>

#include "NBody.h"

#include "utils.h"

inline __device__
float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline __device__
float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __device__
void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline __device__
float4 operator*(float a, float4 b)
{
    return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
}

inline __device__
float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

__global__
void body_sim(float4* pos, float4* vel, int numBodies, float deltaTime, float epsSqr, float4* newPosition, float4* newVelocity)
{
    __shared__ float4 localPos[512];
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int localSize = blockDim.x;

    // Number of tiles we need to iterate
    unsigned int numTiles = numBodies / localSize;

    // position of this work-item
    float4 myPos = pos[gid];

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for(int i = 0; i < numTiles; ++i) {
        // load one tile into local memory
        int idx = i * localSize + tid;
        localPos[tid] = pos[idx];

        // Synchronize to make sure data is available for processing
        __syncthreads();

        // calculate acceleration effect due to each body

        // a[i->j] = m[j] * r[i->j] / (r^2 + epsSqr)^(3/2)

        for(int j = 0; j < localSize; ++j) {
            // Calculate acceleartion caused by particle j on particle i
            float4 r = localPos[j] - myPos;
            float distSqr = r.x * r.x  +  r.y * r.y  +  r.z * r.z;
            float invDist = 1.0f / sqrt(distSqr + epsSqr);
            float invDistCube = invDist * invDist * invDist;
            float s = localPos[j].w * invDistCube;

            // accumulate effect of all particles
            acc += s * r;
        }

        // Synchronize so that next tile can be loaded
        __syncthreads();
    }

    float4 oldVel = vel[gid];

    // updated position and velocity
    float4 newPos = myPos + oldVel * deltaTime + acc * 0.5f * deltaTime * deltaTime;

    newPos.w = myPos.w;

    float4 newVel = oldVel + acc * deltaTime;

    // write to global memory
    newPosition[gid] = newPos;
    newVelocity[gid] = newVel;
}

int numBodies;      /**< No. of particles*/
float* pos;      /**< Output position */
void* me;           /**< Pointing to NBody class */
bool display;

clock_t t1, t2;
int frameCount = 0;
int frameRefCount = 90;
double totalElapsedTime = 0.0;

bool quiet = false;
bool verify = false;
bool timing = true;

float
NBody::random(float randMax, float randMin)
{
    float result;
    result =(float)rand()/(float)RAND_MAX;

    return ((1.0f - result) * randMin + result *randMax);
}

int
NBody::setupNBody()
{
    // make sure numParticles is multiple of group size
    numBodies = numParticles;

    initPos = (float*)malloc(numBodies * sizeof(float4));
    assert(initPos != NULL);
    initVel = (float*)malloc(numBodies * sizeof(float4));
    assert(initVel != NULL);

    /* Create memory objects for position */
    assert(gmacMalloc((void **) &currPos, numBodies * sizeof(float4)) == gmacSuccess);
    assert(gmacMalloc((void **) &currVel, numBodies * sizeof(float4)) == gmacSuccess);

    assert(gmacMalloc((void **) &newPos, numBodies * sizeof(float4)) == gmacSuccess);
    assert(gmacMalloc((void **) &newVel, numBodies * sizeof(float4)) == gmacSuccess);

    pos = currPos;

    /* initialization of inputs */
    for(int i = 0; i < numBodies; ++i)
    {
        int index = 4 * i;

        // First 3 values are position in x,y and z direction
        for(int j = 0; j < 3; ++j)
        {
            currPos[index + j] = random(3, 50);
        }

        // Mass value
        currPos[index + 3] = random(1, 1000);

        // First 3 values are velocity in x,y and z direction
        for(int j = 0; j < 3; ++j)
        {
            currVel[index + j] = 0.0f;
        }

        // unused
        currVel[3] = 0.0f;
    }

    memcpy(initPos, currPos, 4 * numBodies * sizeof(float));
    memcpy(initVel, currVel, 4 * numBodies * sizeof(float));

    return 0;
}

int
NBody::setupCL()
{
    return 0;
}


int
NBody::setupCLKernels()
{
    return 0;
}

int
NBody::runCLKernels()
{
    /*
    * Enqueue a kernel run call.
    */
    body_sim<<<dim3(numBodies/512), dim3(512)>>>(gmacPtr((float4 *) currPos), gmacPtr((float4 *) currVel),
                                                 numBodies, delT, espSqr,
                                                 gmacPtr((float4 *) newPos), gmacPtr((float4 *) newVel));
    gmacThreadSynchronize();

    gmacMemcpy(currPos, newPos, sizeof(float4) * numBodies);
    gmacMemcpy(currVel, newVel, sizeof(float4) * numBodies);
    return 0;
}

/*
* n-body simulation on cpu
*/
void
NBody::nBodyCPUReference()
{
    //Iterate for all samples
    for(int i = 0; i < numBodies; ++i) {
        int myIndex = 4 * i;
        float acc[3] = {0.0f, 0.0f, 0.0f};
        for(int j = 0; j < numBodies; ++j) {
            float r[3];
            int index = 4 * j;

            float distSqr = 0.0f;
            for(int k = 0; k < 3; ++k) {
                r[k] = refPos[index + k] - refPos[myIndex + k];

                distSqr += r[k] * r[k];
            }

            float invDist = 1.0f / sqrt(distSqr + espSqr);
            float invDistCube =  invDist * invDist * invDist;
            float s = refPos[index + 3] * invDistCube;

            for(int k = 0; k < 3; ++k)
            {
                acc[k] += s * r[k];
            }
        }

        for(int k = 0; k < 3; ++k)
        {
            refPos[myIndex + k] += refVel[myIndex + k] * delT + 0.5f * acc[k] * delT * delT;
            refVel[myIndex + k] += acc[k] * delT;
        }
    }
}

int
NBody::setup()
{
    setupNBody();

    setupCL();

    display = !quiet && !verify;

    return 0;
}

/**
* @brief Initialize GL
*/
void
GLInit()
{
    glClearColor(0.0 ,0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
}

/**
* @brief Glut Idle function
*/
void
idle()
{
    glutPostRedisplay();
}

/**
* @brief Glut reshape func
*
* @param w numParticles of OpenGL window
* @param h height of OpenGL window
*/
void
reShape(int w,int h)
{
    glViewport(0, 0, w, h);

    glViewport(0, 0, w, h);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluPerspective(45.0f, w/h, 1.0f, 1000.0f);
    gluLookAt (0.0, 0.0, -2.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0);
}

/**
* @brief OpenGL display function
*/
void displayfunc()
{
    gmactime_t s, t;
    getTime(&s);
    frameCount++;

    glClearColor(0.0 ,0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);

    glPointSize(1.0);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_BLEND);
    glDepthMask(GL_FALSE);

    glColor3f(1.0f,0.6f,0.0f);

    //Calling kernel for calculatig subsequent positions
    ((NBody*)me)->runCLKernels();

    glBegin(GL_POINTS);
    for(int i=0; i < numBodies; ++i)
    {
        //divided by 300 just for scaling
        glVertex3d(pos[i*4+ 0]/300, pos[i*4+1]/300, pos[i*4+2]/300);
    }
    glEnd();

    glFlush();
    glutSwapBuffers();

    getTime(&t);
    totalElapsedTime += (getTimeStamp(t) - getTimeStamp(s)); //(double)(t2 - t1);
    if(frameCount > frameRefCount) {
        // set GLUT Window Title
        char title[256];
        int framesPerSec = (int)(frameCount / (totalElapsedTime / 1e6));
#if defined (_WIN32) && !defined(__MINGW32__)
        sprintf_s(title, 256, "OpenCL NBody | %d fps ", framesPerSec);
#else 
        sprintf(title, "OpenCL NBody | %d fps", framesPerSec);
#endif
        glutSetWindowTitle(title);
        frameCount = 0;
        totalElapsedTime = 0.0;
    }
}

/* keyboard function */
void
keyboardFunc(unsigned char key, int mouseX, int mouseY)
{
    switch(key)
    {
        /* If the user hits escape or Q, then exit */
        /* ESCAPE_KEY = 27 */
    case 27:
    case 'q':
    case 'Q':
        {
                exit(0);
        }
    default:
        break;
    }
}


int
NBody::run()
{
    /* Arguments are set and execution call is enqueued on command buffer */
    setupCLKernels();

    if(verify || timing) {
        for(int i = 0; i < iterations; ++i) {
            runCLKernels();
        }
    }

    return 0;
}

static bool
compareVector(float *a, float *b, unsigned len, float threshold)
{
    for (unsigned i = 0; i < len; i++) {
        if (fabsf(a[i] - b[i]) >= threshold) return false;
    }

    return true;
}

int
NBody::verifyResults()
{
    if(verify) {
        /* reference implementation
        * it overwrites the input array with the output
        */
        refPos = (float*)malloc(numBodies * sizeof(float4));
        assert(refPos != NULL);
        refVel = (float*)malloc(numBodies * sizeof(float4));
        assert(refVel != NULL);

        memcpy(refPos, initPos, 4 * numBodies * sizeof(float));
        memcpy(refVel, initVel, 4 * numBodies * sizeof(float));

        for(int i = 0; i < iterations; ++i) {
            nBodyCPUReference();
        }

        /* compare the results and see if they match */
        if(!compareVector(pos, refPos, 4 * numBodies, 0.00001)) {
            exit(1);
        }
    }

    return 0;
}

void
NBody::printStats()
{
    // TODO Implement timing
#if 0
    std::string strArray[4] =
    {
        "Particles",
        "Iterations",
        "Time(sec)",
        "kernelTime(sec)"
    };

    std::string stats[4];
    totalTime = setupTime + kernelTime;

    stats[0] = sampleCommon->toString(numParticles, std::dec);
    stats[1] = sampleCommon->toString(iterations, std::dec);
    stats[2] = sampleCommon->toString(totalTime, std::dec);
    stats[3] = sampleCommon->toString(kernelTime, std::dec);

    this->SDKSample::printStats(strArray, stats, 4);
#endif
}

int
NBody::cleanup()
{
    /* Releases OpenCL resources (Context, Memory etc.) */
    assert(gmacFree(currPos) == gmacSuccess);
    assert(gmacFree(currVel) == gmacSuccess);
    assert(gmacFree(newPos) == gmacSuccess);
    assert(gmacFree(newVel) == gmacSuccess);

    return 0;
}

NBody::~NBody()
{
    /* release program resources */
    if(initPos) {
        free(initPos);
        initPos = NULL;
    }

    if(initVel) {
        free(initVel);
        initVel = NULL;
    }

    if(refPos) {
        free(refPos);
        refPos = NULL;
    }

    if(refVel) {
        free(refVel);
        refVel = NULL;
    }
}


int
main(int argc, char * argv[])
{
    NBody clNBody("OpenCL NBody");
    me = &clNBody;

    clNBody.setup();
    clNBody.run();
    clNBody.verifyResults();

    clNBody.printStats();

    if(display) {
        // Run in  graphical window if requested
        glutInit(&argc, argv);
        glutInitWindowPosition(100,10);
        glutInitWindowSize(600,600);
        glutInitDisplayMode( GLUT_RGB | GLUT_DOUBLE );
        glutCreateWindow("nbody simulation");
        GLInit();
        glutDisplayFunc(displayfunc);
        glutReshapeFunc(reShape);
        glutIdleFunc(idle);
        glutKeyboardFunc(keyboardFunc);
        glutMainLoop();
    }

    clNBody.cleanup();

    return 0;
}
