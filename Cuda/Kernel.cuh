#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/CudaHelper.h"
#include <Eigen/Dense>
#include <vector>

const unsigned blockSize = 32; // Threads per block

__global__ void handleParticleKernel(unsigned numParticles, float radius, float deltaTime, Particle* particles, bool GRAVITY, float gravity);

__global__ void handleParticleKernelOpt(unsigned numParticles, float radius, float deltaTime, Particle* particles, bool GRAVITY, float gravity);


void launchParticleKernel(CudaHelper& cudaHelper, float deltaTime);

void launchParticleKernelOpt(CudaHelper& cudaHelper, float deltaTime);
