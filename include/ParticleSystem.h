#pragma once
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include "Particle.h"


class ParticleSystem {
private:
	// Constants
	const float GRAVITY		= 9.8f;
	const float DAMPENER	= 0.7f;
	const unsigned SUBSTEPS = 8;

	// Particle System state flags
	bool m_RUNNING = true;
	bool m_GRAVITY = false; // True = gravity acting on particles
	bool m_VERLET  = true;

	// MISC particle information
	unsigned int m_numParticles;
	unsigned int m_WindowSize;

	// Particle containers
	std::vector<Particle> m_particles;
	
	/*float* m_particlePos;
	float* m_particleColor;*/
	std::vector<float> m_particlePos;
	std::vector<float> m_particleColor;

	float m_radius;

public:
	// Constructors/destructors
	ParticleSystem(unsigned numParticles, unsigned windowSize, float radius);
	ParticleSystem(unsigned numParticles, unsigned windowSize, float radius, bool random);
	~ParticleSystem();


	// Set and get particle system state
	float* getParticlePos();
	float* getParticleColor();
	bool isRunning();
	void toggleRunning();
	void toggleGravity();

	// Particle System logic
	void simulate(float deltaTime);
	void handleParticleMovement(float deltaTime);
	void handleCollisions();

	// Debugging tools
	void printParticlePos();
	void printParticleVel();
};

