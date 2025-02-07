#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/CudaHelper.h"
#include <Eigen/Dense>
#include <vector>

const unsigned blockSize = 32; // Threads per block

__global__ void handleCollisionsKernel(unsigned numParticles, Particle* particles, bool GRAVITY);

__global__ void handleMovementKernel(unsigned numParticles, float deltaTime, float* particlePos, float* particleVel, bool GRAVITY);

__global__ void handleMovementKernelEigen(unsigned numParticles, float deltaTime, Particle* particles, bool GRAVITY);



void launchMovementKernel(CudaHelper& cudaHelper, float deltaTime);

void launchMovementKernelEigen(CudaHelper& cudaHelper, float deltaTime);

void launchCollisionsKernel(CudaHelper& cudaHelper, float deltaTime);
