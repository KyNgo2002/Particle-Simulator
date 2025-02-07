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

CudaHelper::CudaHelper() 
    : h_particles (nullptr), d_particles (nullptr), m_GRAVITY (false), m_numParticles (0), m_radius (0.0f) {
}

CudaHelper::CudaHelper(unsigned numParticles, float radius, bool gravity, Particle* particles) 
    : m_numParticles(numParticles), m_radius(radius), m_GRAVITY(gravity), h_particles (particles) {

    // Allocate space on the GPU
    cudaMalloc(&d_particles, m_numParticles * sizeof(Particle));
    cudaCheckErrors("Malloc failure: Particles");

    // Memory copy: Host to device
    cudaMemcpy(d_particles, h_particles, numParticles * sizeof(Particle), cudaMemcpyHostToDevice);
    cudaCheckErrors("Memcpy failure -> Particle collisions host to device");
}

CudaHelper::~CudaHelper() {

}

void CudaHelper::clean() {
    cudaFree(d_particles);
    cudaCheckErrors("Free failure: Particles");
}


