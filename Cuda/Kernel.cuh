#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const unsigned blockSize = 32; // Threads per block

__global__ void handleMovementKernel(unsigned numParticles, float deltaTime, float* particlePos, float* particleVel, bool GRAVITY);

void launchMovementKernel(unsigned numParticles, float deltaTime, float* h_particlePos, float* h_particleVel, bool m_GRAVITY);