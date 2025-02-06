#include <stdlib.h>
#include <stdio.h>
#include "../include/CudaHelper.h"

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

CudaHelper::CudaHelper() {
}

CudaHelper::CudaHelper(unsigned numParticles, bool gravity, float* h_particlePos, float* h_particleVel) : m_numParticles( numParticles ), m_GRAVITY( gravity ), d_particlePos( nullptr ), d_particleVel( nullptr )  {
    this->h_particlePos = h_particlePos;
    this->h_particleVel = h_particleVel;
    // Memory allocation: Device
    cudaMalloc(&d_particlePos, numParticles * 2 * sizeof(float));
    cudaCheckErrors("Malloc failure: Particle Positions");
    cudaMalloc(&d_particleVel, numParticles * 2 * sizeof(float));
    cudaCheckErrors("Malloc failure: Particle Velocities");
}

CudaHelper::~CudaHelper() {
    
}

void CudaHelper::clean() {
    cudaFree(d_particlePos);
    cudaCheckErrors("Free failure: Particle Positions");
    cudaFree(d_particleVel);
    cudaCheckErrors("Free failure: Particle Velocities");
}


