#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/CudaHelper.h"

const unsigned blockSize = 32; // Threads per block

__global__ void handleMovementKernel(unsigned numParticles, float deltaTime, float* particlePos, float* particleVel, bool GRAVITY);

void launchMovementKernel(CudaHelper& cudaHelper, float deltaTime);