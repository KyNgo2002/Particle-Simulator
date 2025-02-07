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

CudaHelper::CudaHelper() : h_particlePos (nullptr), h_particleVel (nullptr), d_particlePos (nullptr), d_particleVel (nullptr), h_particles (nullptr), d_particles (nullptr) {
    m_GRAVITY = false;
    m_numParticles = 0;
}

CudaHelper::CudaHelper(unsigned numParticles, bool gravity, float* h_particlePos, float* h_particleVel) : m_numParticles( numParticles ), m_GRAVITY( gravity ), d_particlePos( nullptr ), d_particleVel( nullptr )  {
    this->h_particlePos = h_particlePos;
    this->h_particleVel = h_particleVel;
    this->h_particles = nullptr;
    this->d_particles = nullptr;


    // Memory allocation: Device
    cudaMalloc(&d_particlePos, numParticles * 2 * sizeof(float));
    cudaCheckErrors("Malloc failure: Particle Positions");
    cudaMalloc(&d_particleVel, numParticles * 2 * sizeof(float));
    cudaCheckErrors("Malloc failure: Particle Velocities");
}

CudaHelper::CudaHelper(unsigned numParticles, bool gravity, Particle* h_particles) : m_numParticles(numParticles), m_GRAVITY(gravity), d_particlePos(nullptr), d_particleVel(nullptr) {
    this->h_particlePos = nullptr;
    this->h_particleVel = nullptr;
    this->h_particles = h_particles;

    cudaMalloc(&d_particles, m_numParticles * sizeof(Particle));
    cudaCheckErrors("Malloc failure: Particles");
}

CudaHelper::~CudaHelper() {
    
}

void CudaHelper::clean() {
    cudaFree(d_particlePos);
    cudaCheckErrors("Free failure: Particle Positions");
    cudaFree(d_particleVel);
    cudaCheckErrors("Free failure: Particle Velocities");
    cudaFree(d_particles);
    cudaCheckErrors("Free failure: Particles");

}


