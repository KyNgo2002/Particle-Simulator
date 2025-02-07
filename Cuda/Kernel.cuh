#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/CudaHelper.h"
#include <Eigen/Dense>
#include <vector>

const unsigned blockSize = 32; // Threads per block


__global__ void handleMovementKernelEigen(unsigned numParticles, float deltaTime, Particle* particles, bool GRAVITY);

__global__ void handleCollisionsKernelEigen(unsigned numParticles, float radius, Particle* particles, bool GRAVITY);

__global__ void handleBothKernelEigen(unsigned numParticles, float radius, float deltaTime, Particle* particles, bool GRAVITY);


void launchMovementKernelEigen(CudaHelper& cudaHelper, float deltaTime);

void launchCollisionsKernelEigen(CudaHelper& cudaHelper);

void launchBothKernelEigen(CudaHelper& cudaHelper, float deltaTime);