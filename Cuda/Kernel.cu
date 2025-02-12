#include "Kernel.cuh"
#include <iostream>

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

__global__ void handleParticleKernel(unsigned numParticles, float radius, float deltaTime, Particle* particles, bool GRAVITY, float gravity) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numParticles) {
        // Movement
        if (GRAVITY) {
            particles[tid].m_velocity[1] -= gravity * deltaTime;
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

__global__ void handleParticleKernelOpt(unsigned numParticles, float radius, float deltaTime, Particle* particles, bool GRAVITY, float gravity) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (GRAVITY) {
        particles[tid].m_velocity[1] -= gravity * deltaTime;
    }
    particles[tid].m_position += particles[tid].m_velocity * deltaTime;

    // Collisions
    for (unsigned i = tid + 1; i < numParticles; ++i) {
        Eigen::Vector2f normal = particles[tid].m_position - particles[i].m_position;
        // Optimization
        if (abs(normal[0]) > 2.1f * radius) continue;
        if (abs(normal[1]) > 2.1f * radius) continue;

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

    bool top = (particles[tid].m_position[1] > (1.0f - radius));
    bool bottom = (particles[tid].m_position[1] < (-1.0f + radius));

    if (top || bottom) {
        particles[tid].m_position[1] = top ? (1.0f - radius) : (-1.0f + radius);
        particles[tid].m_velocity[1] *= -1.0f;
        if (GRAVITY)
            particles[tid].m_velocity[1] *= 0.7;
    }

    bool left = (particles[tid].m_position[0] < (-1.0f + radius));
    bool right = (particles[tid].m_position[0] > (1.0f - radius));
    if (left || right) {
        particles[tid].m_position[0] = left ? (-1.0f + radius) : (1.0f - radius);
        particles[tid].m_velocity[0] *= -1.0f;
        if (GRAVITY)
            particles[tid].m_velocity[0] *= 0.7;
    }
}

void launchParticleKernel(CudaHelper& cudaHelper, float deltaTime) {
    unsigned numParticles = cudaHelper.m_numParticles;
    unsigned numBlocks = (numParticles + blockSize - 1) / blockSize;
    float gravity = 10.0f;

    // Kernel Launch
    handleParticleKernel <<< numBlocks, blockSize >>> (numParticles, cudaHelper.m_radius, deltaTime, cudaHelper.d_particles, cudaHelper.m_GRAVITY, gravity);
    cudaDeviceSynchronize();
    cudaCheckErrors("Kernel Launch -> Handle Particle collisions");

    // Memory copy: Device to host
    cudaMemcpy(cudaHelper.h_particles, cudaHelper.d_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Memcpy failure -> Particle collisions device to host");
}

void launchParticleKernelOpt(CudaHelper& cudaHelper, float deltaTime) {
    unsigned numParticles = cudaHelper.m_numParticles;
    unsigned numBlocks = (numParticles + blockSize - 1) / blockSize;
    float gravity = 10.0f;

    // Kernel Launch
    handleParticleKernelOpt << < numBlocks, blockSize >> > (numParticles, cudaHelper.m_radius, deltaTime, cudaHelper.d_particles, cudaHelper.m_GRAVITY, gravity);
    cudaDeviceSynchronize();
    cudaCheckErrors("Kernel Launch -> Handle Particle collisions");

    // Memory copy: Device to host
    cudaMemcpy(cudaHelper.h_particles, cudaHelper.d_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Memcpy failure -> Particle collisions device to host");
}



