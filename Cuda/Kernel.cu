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

__global__ void handleMovementKernelEigen(unsigned numParticles, float deltaTime, Particle* particles, bool GRAVITY) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numParticles) {
        if (GRAVITY) {
            particles[tid].m_position[1] -= 10.0f * deltaTime;
        }
        particles[tid].m_position += particles[tid].m_velocity * deltaTime;
    }
}

__global__ void handleCollisionsKernelEigen(unsigned numParticles, float radius, Particle* particles, bool GRAVITY) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numParticles) {
        for (unsigned i = tid + 1; i < numParticles; ++i) {
            Eigen::Vector2f normal = particles[tid].m_position - particles[i].m_position;
            // Optimization
            if (abs(normal[0]) > 2.3f * radius) continue;
            if (abs(normal[1]) > 2.3f * radius) continue;

            float dist = normal.norm();

            // Collision occured
            if (dist < 2.0f * radius) {
                Eigen::Vector2f unitNormal = normal.normalized();

                // Position update
                float delta = 0.5f * (2.0f * radius - dist);
                particles[tid].m_position += delta * unitNormal;
                particles[i].m_position -= delta * unitNormal;

                // Velocity Update
                Eigen::Vector2f v1n = (particles[tid].m_velocity.dot(normal)) / (dist * dist) * normal;
                Eigen::Vector2f v1t = (particles[tid].m_velocity - v1n);
                Eigen::Vector2f v2n = (particles[i].m_velocity.dot(normal)) / (dist * dist) * normal;
                Eigen::Vector2f v2t = (particles[i].m_velocity - v2n);

                particles[tid].m_velocity = (v2n + v1t);
                particles[i].m_velocity = (v1n + v2t);

                if (GRAVITY) {
                    particles[tid].m_velocity *= 0.7;
                    particles[i].m_velocity *= 0.7;
                }
            }
        }
        
        // Collision with top border
        if (particles[tid].m_position[1] > (1.0f - radius)) {
            particles[tid].m_position[1] = (1.0f - radius);
            particles[tid].m_velocity[1] *= -1.0f;
            if (GRAVITY)
                particles[tid].m_velocity[1] *= 0.7;
        }
        // Collision with bottom border
        if (particles[tid].m_position[1] < (-1.0f + radius)) {
            particles[tid].m_position[1] = (-1.0f + radius);
            particles[tid].m_velocity[1] *= -1.0f;
            if (GRAVITY)
                particles[tid].m_velocity[1] *= 0.7;
        }
        // Collision with left border
        if (particles[tid].m_position[0] < (-1.0f + radius)) {
            particles[tid].m_position[0] = (-1.0f + radius);
            particles[tid].m_velocity[0] *= -1.0f;
            if (GRAVITY)
                particles[tid].m_velocity[0] *= 0.7;
        }
        // Collision with right border
        if (particles[tid].m_position[0] > (1.0f - radius)) {
            particles[tid].m_position[0] = (1.0f - radius);
            particles[tid].m_velocity[0] *= -1.0f;
            if (GRAVITY)
                particles[tid].m_velocity[0] *= 0.7;
        }
    }
}

__global__ void handleBothKernelEigen(unsigned numParticles, float radius, float deltaTime, Particle* particles, bool GRAVITY) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numParticles) {
        // Movement
        if (GRAVITY) {
            particles[tid].m_position[1] -= 10.0f * deltaTime;
        }
        particles[tid].m_position += particles[tid].m_velocity * deltaTime;

        // Collisions
        for (unsigned i = tid + 1; i < numParticles; ++i) {
            Eigen::Vector2f normal = particles[tid].m_position - particles[i].m_position;
            // Optimization
            if (abs(normal[0]) > 2.3f * radius) continue;
            if (abs(normal[1]) > 2.3f * radius) continue;

            float dist = normal.norm();

            // Collision occured
            if (dist < 2.0f * radius) {
                Eigen::Vector2f unitNormal = normal.normalized();

                // Position update
                float delta = 0.5f * (2.0f * radius - dist);
                particles[tid].m_position += delta * unitNormal;
                particles[i].m_position -= delta * unitNormal;

                // Velocity Update
                Eigen::Vector2f v1n = (particles[tid].m_velocity.dot(normal)) / (dist * dist) * normal;
                Eigen::Vector2f v1t = (particles[tid].m_velocity - v1n);
                Eigen::Vector2f v2n = (particles[i].m_velocity.dot(normal)) / (dist * dist) * normal;
                Eigen::Vector2f v2t = (particles[i].m_velocity - v2n);

                particles[tid].m_velocity = (v2n + v1t);
                particles[i].m_velocity = (v1n + v2t);

                if (GRAVITY) {
                    particles[tid].m_velocity *= 0.7;
                    particles[i].m_velocity *= 0.7;
                }
            }
        }

        // Collision with top border
        if (particles[tid].m_position[1] > (1.0f - radius)) {
            particles[tid].m_position[1] = (1.0f - radius);
            particles[tid].m_velocity[1] *= -1.0f;
            if (GRAVITY)
                particles[tid].m_velocity[1] *= 0.7;
        }
        // Collision with bottom border
        if (particles[tid].m_position[1] < (-1.0f + radius)) {
            particles[tid].m_position[1] = (-1.0f + radius);
            particles[tid].m_velocity[1] *= -1.0f;
            if (GRAVITY)
                particles[tid].m_velocity[1] *= 0.7;
        }
        // Collision with left border
        if (particles[tid].m_position[0] < (-1.0f + radius)) {
            particles[tid].m_position[0] = (-1.0f + radius);
            particles[tid].m_velocity[0] *= -1.0f;
            if (GRAVITY)
                particles[tid].m_velocity[0] *= 0.7;
        }
        // Collision with right border
        if (particles[tid].m_position[0] > (1.0f - radius)) {
            particles[tid].m_position[0] = (1.0f - radius);
            particles[tid].m_velocity[0] *= -1.0f;
            if (GRAVITY)
                particles[tid].m_velocity[0] *= 0.7;
        }
    }
}

void launchMovementKernelEigen(CudaHelper& cudaHelper, float deltaTime) {

    unsigned numParticles = cudaHelper.m_numParticles;
    unsigned numBlocks = (numParticles + blockSize - 1) / blockSize;

    // Memory copy: Host to device
    cudaMemcpy(cudaHelper.d_particles, cudaHelper.h_particles, numParticles * sizeof(Particle), cudaMemcpyHostToDevice);
    cudaCheckErrors("Memcpy failure -> Particle Positions host to device");

    // Kernel Launch
    handleMovementKernelEigen <<< numBlocks, blockSize >>> (numParticles, deltaTime, cudaHelper.d_particles, cudaHelper.m_GRAVITY);
    cudaDeviceSynchronize();
    cudaCheckErrors("Kernel Launch -> Calculate position kernel");

    // Memory copy: Device to host
    cudaMemcpy(cudaHelper.h_particles, cudaHelper.d_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Memcpy failure -> Particle Positions device to host");

}

void launchCollisionsKernelEigen(CudaHelper& cudaHelper) {

    unsigned numParticles = cudaHelper.m_numParticles;
    unsigned numBlocks = (numParticles + blockSize - 1) / blockSize;

    // Memory copy: Host to device
    cudaMemcpy(cudaHelper.d_particles, cudaHelper.h_particles, numParticles * sizeof(Particle), cudaMemcpyHostToDevice);
    cudaCheckErrors("Memcpy failure -> Particle collisions host to device");

    // Kernel Launch
    handleCollisionsKernelEigen <<< numBlocks, blockSize >>> (numParticles, cudaHelper.m_radius, cudaHelper.d_particles, cudaHelper.m_GRAVITY);
    cudaDeviceSynchronize();
    cudaCheckErrors("Kernel Launch -> Handle Particle collisions");

    // Memory copy: Device to host
    cudaMemcpy(cudaHelper.h_particles, cudaHelper.d_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Memcpy failure -> Particle collisions device to host");

}

void launchBothKernelEigen(CudaHelper& cudaHelper, float deltaTime) {
    unsigned numParticles = cudaHelper.m_numParticles;
    unsigned numBlocks = (numParticles + blockSize - 1) / blockSize;

    // Memory copy: Host to device
    cudaMemcpy(cudaHelper.d_particles, cudaHelper.h_particles, numParticles * sizeof(Particle), cudaMemcpyHostToDevice);
    cudaCheckErrors("Memcpy failure -> Particle collisions host to device");

    // Kernel Launch
    handleBothKernelEigen <<< numBlocks, blockSize >>> (numParticles, cudaHelper.m_radius, deltaTime, cudaHelper.d_particles, cudaHelper.m_GRAVITY);
    cudaDeviceSynchronize();
    cudaCheckErrors("Kernel Launch -> Handle Particle collisions");

    // Memory copy: Device to host
    cudaMemcpy(cudaHelper.h_particles, cudaHelper.d_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Memcpy failure -> Particle collisions device to host");
}



