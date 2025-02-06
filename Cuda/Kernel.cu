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

void launchMovementKernel(CudaHelper& cudaHelper, float deltaTime) {

	unsigned numBlocks = (cudaHelper.m_numParticles + blockSize - 1) / blockSize; 

    // Memory copy: Host to device
    cudaMemcpy(cudaHelper.d_particlePos, cudaHelper.h_particlePos, cudaHelper.m_numParticles * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("Memcpy failure: Particle Positions host to device");
    cudaMemcpy(cudaHelper.d_particleVel, cudaHelper.h_particleVel, cudaHelper.m_numParticles * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("Memcpy failure: Particle Velocities host to device");

    // Kernel Launch
	handleMovementKernel <<< numBlocks, blockSize >>> (cudaHelper.m_numParticles, deltaTime, cudaHelper.d_particlePos, cudaHelper.d_particleVel, cudaHelper.m_GRAVITY);
    cudaDeviceSynchronize();
    cudaCheckErrors("Kernel Launch: Calculate position kernel");

    // Memory copy: Device to host
    cudaMemcpy(cudaHelper.h_particlePos, cudaHelper.d_particlePos, cudaHelper.m_numParticles * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Memcpy failure: Particle Positions device to host");
    cudaMemcpy(cudaHelper.h_particleVel, cudaHelper.d_particleVel, cudaHelper.m_numParticles * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Memcpy failure: Particle Velocities device to host");

}