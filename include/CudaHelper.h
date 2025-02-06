#pragma once
#include <cuda_runtime.h>

class CudaHelper {
public:
	unsigned m_numParticles;
	bool m_GRAVITY;
	float* h_particlePos;
	float* h_particleVel;
	float* d_particlePos;
	float* d_particleVel;
	
	CudaHelper();
    CudaHelper(unsigned numParticles, bool gravity, float* h_particlePos, float* h_particleVel);
	~CudaHelper();
	void clean();
};

