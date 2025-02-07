#pragma once
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include "Particle.h"
#include "../Cuda/Kernel.cuh"
#include <cuda_runtime.h>
#include "CudaHelper.h"

class ParticleSystem {
private:
	// Constants
	const float GRAVITY		= 10.0f;
	const float DAMPENER	= 0.7f;
	const unsigned SUBSTEPS = 8;

	// Particle System state flags
	bool m_RUNNING = true;
	bool m_GRAVITY = false; // True = gravity acting on particles
	bool m_VERLET  = true;
	bool m_CUDA_ENABLED = false;
	bool m_EIGEN_ENABLED = false;

	// MISC particle information
	unsigned int m_numParticles;
	unsigned int m_WindowSize;

	// Particle containers
	std::vector<Particle> m_particles;
	
	std::vector<float> m_particlePos;
	std::vector<float> m_particleVel;
	std::vector<float> m_particleColor;

	float m_radius;

	CudaHelper cudaHelper;

public:
	// Constructors/destructors
	ParticleSystem(unsigned numParticles, unsigned windowSize, float radius, bool enableCuda);
	~ParticleSystem();

	// Set and get particle system state
	float* getParticlePos();
	float* getParticleColor();
	bool isRunning() const;
	void toggleRunning();
	void toggleGravity();

	// Particle System logic
	void simulate(float deltaTime);
	void handleMovement(float deltaTime);
	void handleCollisions();

	// Debugging tools
	void printParticlePos();
	void printParticleVel();
};

