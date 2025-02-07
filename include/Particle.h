#pragma once
#include <Eigen/Dense>
#include <vector>

class Particle {
public:
	// Particle information
	Eigen::Vector2f m_position;
	Eigen::Vector2f m_velocity;

	// Constructor
	Particle(float x, float y, float vx, float vy);

	// Particle system logic
	void updateMovement(float deltaTime, bool verlet, bool gravity);
};

