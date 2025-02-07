#include "../include/Particle.h"

/// Initializes position, velocity, acceleration, and color of particle
Particle::Particle(float x, float y, float vx, float vy) {
	m_position[0] = x;
	m_position[1] = y;
	m_velocity[0] = vx;
	m_velocity[1] = vy;
}

/// Handles movement of particle
void Particle::updateMovement(float deltaTime, bool verlet, bool gravity) {
	if (gravity)
		m_velocity[1] -= gravity * deltaTime;
	m_position += m_velocity * deltaTime;
}

