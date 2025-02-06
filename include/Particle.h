#pragma once
#include <Eigen/Dense>
#include <vector>

class Particle {
public:
	// Particle information

	Eigen::Vector2f m_position;
	Eigen::Vector2f m_previousPosition;
	Eigen::Vector2f m_velocity;
	Eigen::Vector2f m_acceleration;
	Eigen::Vector3f m_color;

	// Constructor
	Particle(float x, float y, float vx, float vy, float r, float g, float b);

	// Particle system logic
	void updateMovement(float deltaTime, bool verlet, bool gravity);
	void accelerate(Eigen::Vector2f& acc);

};

