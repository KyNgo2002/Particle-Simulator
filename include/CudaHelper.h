#pragma once
#include <cuda_runtime.h>
#include <Eigen/dense>
#include <vector>
#include "Particle.h"

class CudaHelper {
public:
	unsigned m_numParticles;
	bool m_GRAVITY;
	float* h_particlePos;
	float* h_particleVel;
	float* d_particlePos;
	float* d_particleVel;
	Particle* h_particles;
	Particle* d_particles;


	
	CudaHelper();
    CudaHelper(unsigned numParticles, bool gravity, float* h_particlePos, float* h_particleVel);
	CudaHelper(unsigned numParticles, bool gravity, Particle* h_particles);

	~CudaHelper();
	void clean();
};

