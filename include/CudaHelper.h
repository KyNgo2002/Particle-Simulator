#pragma once
#include <cuda_runtime.h>
#include <Eigen/dense>
#include <vector>
#include "Particle.h"

class CudaHelper {
public:
	unsigned m_numParticles;
	float m_radius;

	bool m_GRAVITY;

	Particle* h_particles;
	Particle* d_particles;

	
	CudaHelper();
	CudaHelper(unsigned numParticles, float radius, bool gravity, Particle* h_particles);

	~CudaHelper();
	void clean();
};

