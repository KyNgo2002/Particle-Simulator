#include "Kernel.cuh"

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

__global__ void handleMovementKernel(unsigned numParticles, float deltaTime, float* particlePos, float* particleVel, bool GRAVITY) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numParticles) {
        if (GRAVITY) {
            particleVel[tid * 2 + 1] -= 10.0f * deltaTime;
        }
        particlePos[tid * 2] += particleVel[tid * 2] * deltaTime;
        particlePos[tid * 2 + 1] += particleVel[tid * 2 + 1] * deltaTime;
    }
}

void launchMovementKernel(unsigned numParticles, float deltaTime, float* h_particlePos, float* h_particleVel, bool GRAVITY) {

	unsigned numBlocks = (numParticles + blockSize - 1) / blockSize; 

	// Kernel launch preprocessing
    float* d_particlePos;
    float* d_particleVel;

    // Memory allocation: Device
    cudaMalloc(&d_particlePos, numParticles * 2 * sizeof(float));
    cudaCheckErrors("Malloc failure: Particle Positions");
    cudaMalloc(&d_particleVel, numParticles * 2 * sizeof(float));
    cudaCheckErrors("Malloc failure: Particle Velocities");

    // Memory copy: Host to device
    cudaMemcpy(d_particlePos, h_particlePos, numParticles * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("Memcpy failure: Particle Positions host to device");
    cudaMemcpy(d_particleVel, h_particleVel, numParticles * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("Memcpy failure: Particle Velocities host to device");

    // Kernel Launch
	handleMovementKernel <<< numBlocks, blockSize >>> (numParticles, deltaTime, d_particlePos, d_particleVel, GRAVITY);
    cudaDeviceSynchronize();
    cudaCheckErrors("Kernel Launch: Calculate position kernel");

    // Memory copy: Device to host
    cudaMemcpy(h_particlePos, d_particlePos, numParticles * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Memcpy failure: Particle Positions device to host");
    cudaMemcpy(h_particleVel, d_particleVel, numParticles * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Memcpy failure: Particle Velocities device to host");

    cudaFree(d_particlePos);
    cudaCheckErrors("Free failure: Particle Positions");
    cudaFree(d_particleVel);
    cudaCheckErrors("Free failure: Particle Velocities");


}