#include "../include/Particle.h"

/// Initializes position, velocity, acceleration, and color of particle
Particle::Particle(float x, float y, float vx, float vy, float r, float g, float b) {
	m_position[0] = x;
	m_position[1] = y;
	m_previousPosition[0] = x;
	m_previousPosition[1] = y;
	m_velocity[0] = vx;
	m_velocity[1] = vy;
	m_acceleration[0] = 0.0f;
	m_acceleration[1] = -9.8f;
	m_color[0] = r;
	m_color[1] = g;
	m_color[2] = b;
}

/// Handles movement of particle
void Particle::updateMovement(float deltaTime, bool verlet, bool gravity) {
	if (verlet) {
		
	}
	else {
		if (gravity)
			m_velocity[1] -= gravity * deltaTime;
		m_position += m_velocity * deltaTime;
	}
}

void Particle::accelerate(Eigen::Vector2f& acc) {
	m_acceleration += acc;
}
