#include "../include/ParticleSystem.h"

// Initializes particles with position, velocity, and color
ParticleSystem::ParticleSystem(unsigned numParticles, unsigned windowSize, float radius, bool enableCuda)
	: m_numParticles{ numParticles }, m_WindowSize{ windowSize }, m_radius{ radius }, m_CUDA_ENABLED{ enableCuda } {

	m_particlePos = std::vector<float>(m_numParticles * 2);
	m_particleVel = std::vector<float>(m_numParticles * 2);
	m_particleColor = std::vector<float>(m_numParticles * 3);

	m_particles.reserve(numParticles);

	int count = m_numParticles;
	int rows = m_numParticles / 10 + (m_numParticles % 10 != 0);
	float x = 0.0f;
	float y = 0.0f;
	for (int i = 0; i < rows; ++i) {
		int cols = std::min(10, count);
		for (int j = 0; j < cols; ++j) {
			x = ((cols / 2.0f - j) * 50.0f) / windowSize;
			y = ((rows / 2.0f - i) * 50.0f) / windowSize;

			float vx = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f) * 2.0f;
			float vy = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f) * 2.0f;

			float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			float g = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			float b = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			
			m_particles.push_back(Particle(x, y, vx, vy));

			m_particlePos[i * 20 + j * 2] = x;
			m_particlePos[i * 20 + j * 2 + 1] = y;
			m_particleVel[i * 20 + j * 2] = vx;
			m_particleVel[i * 20 + j * 2 + 1] = vy;
			m_particleColor[i * 30 + j * 3] = r;
			m_particleColor[i * 30 + j * 3 + 1] = g;
			m_particleColor[i * 30 + j * 3 + 2] = b;

			count--;
		}
	}

	if (m_CUDA_ENABLED)
		cudaHelper = CudaHelper(m_numParticles, m_radius, m_GRAVITY, m_particles.data());
}

// Cleans up particle system
ParticleSystem::~ParticleSystem() {
	cudaHelper.clean();
}

// Controls particle movement and collision calculations 
void ParticleSystem::simulate(float deltaTime){
	float subDeltaTime = deltaTime / SUBSTEPS;
	for (unsigned i = 0; i < SUBSTEPS; ++i) {
		if (m_CUDA_ENABLED) {
			launchBothKernelEigen(cudaHelper, subDeltaTime);
			for (unsigned j = 0; j < m_numParticles; ++j) {
				m_particlePos[j * 2] = m_particles[j].m_position[0];
				m_particlePos[j * 2 + 1] = m_particles[j].m_position[1];
			}
		}
		else {
			handleMovement(subDeltaTime);
			handleCollisions();
		}
	}
}

// Handles particle movement
void ParticleSystem::handleMovement(float deltaTime) {
	for (unsigned i = 0; i < m_numParticles; ++i) {
		if (m_GRAVITY) {
			m_particles[i].m_velocity[1] -= GRAVITY * deltaTime;
			m_particleVel[i * 2 + 1] -= GRAVITY * deltaTime;
		}
		m_particles[i].m_position += m_particles[i].m_velocity * deltaTime;
		m_particlePos[i * 2] = m_particles[i].m_position[0];
		m_particlePos[i * 2 + 1] = m_particles[i].m_position[1];
	}
}

// Handles particle collisions
void ParticleSystem::handleCollisions() {
	// Particle->Particle Collision
	for (unsigned i = 0; i < m_numParticles; ++i) {
		for (unsigned j = i + 1; j < m_numParticles; ++j) {
			Eigen::Vector2f normal = m_particles[i].m_position - m_particles[j].m_position;
			// Optimization
			if (abs(normal[0]) > 2.3f * m_radius) continue;
			if (abs(normal[1]) > 2.3f * m_radius) continue;

			float dist = normal.norm();

			// Collision occured
			if (dist < 2.0f * m_radius) {
				Eigen::Vector2f unitNormal = normal.normalized();

				// Position update
				float delta = 0.5f * (2.0f * m_radius - dist);
				m_particles[i].m_position += delta * unitNormal;
				m_particlePos[i * 2] = m_particles[i].m_position[0];
				m_particlePos[i * 2 + 1] = m_particles[i].m_position[1];

				m_particles[j].m_position -= delta * unitNormal;
				m_particlePos[j * 2] = m_particles[j].m_position[0];
				m_particlePos[j * 2 + 1] = m_particles[j].m_position[1];

				// Velocity Update
				Eigen::Vector2f v1n = (m_particles[i].m_velocity.dot(normal)) / (dist * dist) * normal;
				Eigen::Vector2f v1t = (m_particles[i].m_velocity - v1n);
				Eigen::Vector2f v2n = (m_particles[j].m_velocity.dot(normal)) / (dist * dist) * normal;
				Eigen::Vector2f v2t = (m_particles[j].m_velocity - v2n);

				m_particles[i].m_velocity = (v2n + v1t);
				m_particles[j].m_velocity = (v1n + v2t);
				m_particleVel[i * 2] = m_particles[i].m_velocity[0];
				m_particleVel[i * 2 + 1] = m_particles[i].m_velocity[1];
				m_particleVel[j * 2] = m_particles[j].m_velocity[0];
				m_particleVel[j * 2 + 1] = m_particles[j].m_velocity[1];

				if (m_GRAVITY) {
					m_particles[i].m_velocity *= DAMPENER;
					m_particles[j].m_velocity *= DAMPENER;
					m_particleVel[i * 2] *= DAMPENER;
					m_particleVel[i * 2 + 1] *= DAMPENER;
					m_particleVel[j * 2] *= DAMPENER;
					m_particleVel[j * 2 + 1] *= DAMPENER;
				}
			}
		}
	}

	for (unsigned i = 0; i < m_numParticles; ++i) {
		// Collision with top border
		if (m_particles[i].m_position[1] > (1.0f - m_radius)) {

			m_particles[i].m_position[1] = (1.0f - m_radius);
			m_particlePos[i * 2 + 1] = m_particles[i].m_position[1];
			m_particles[i].m_velocity[1] *= -1.0f;
			m_particleVel[i * 2 + 1] = m_particles[i].m_velocity[1];

			if (m_GRAVITY)
				m_particles[i].m_velocity[1] *= DAMPENER;
		}
		// Collision with bottom border
		if (m_particles[i].m_position[1] < (-1.0f + m_radius)) {

			m_particles[i].m_position[1] = (-1.0f + m_radius);
			m_particlePos[i * 2 + 1] = m_particles[i].m_position[1];
			m_particles[i].m_velocity[1] *= -1.0f;
			m_particleVel[i * 2 + 1] = m_particles[i].m_velocity[1];

			if (m_GRAVITY)
				m_particles[i].m_velocity[1] *= DAMPENER;
		}
		// Collision with left border
		if (m_particles[i].m_position[0] < (-1.0f + m_radius)) {

			m_particles[i].m_position[0] = (-1.0f + m_radius);
			m_particlePos[i * 2] = m_particles[i].m_position[0];
			m_particles[i].m_velocity[0] *= -1.0f;
			m_particleVel[i * 2] = m_particles[i].m_velocity[0];

			if (m_GRAVITY)
				m_particles[i].m_velocity[0] *= DAMPENER;
		}
		// Collision with right border
		if (m_particles[i].m_position[0] > (1.0f - m_radius)) {

			m_particles[i].m_position[0] = (1.0f - m_radius);
			m_particlePos[i * 2] = m_particles[i].m_position[0];
			m_particles[i].m_velocity[0] *= -1.0f;
			m_particleVel[i * 2] = m_particles[i].m_velocity[0];

			if (m_GRAVITY)
				m_particles[i].m_velocity[0] *= DAMPENER;
		}
	}
}

// Returns array of particle positons
float* ParticleSystem::getParticlePos() {
	return m_particlePos.data();
}

// Gets array of particle colors
float* ParticleSystem::getParticleColor() {
	return m_particleColor.data();
}

// Returns whether or not the particle system is running
bool ParticleSystem::isRunning() const {
	return m_RUNNING;
}

// Toggles the particle system on and off
void ParticleSystem::toggleRunning() {
	m_RUNNING = !m_RUNNING;
}

// Toggles Gravity
void ParticleSystem::toggleGravity() {
	m_GRAVITY = !m_GRAVITY;
	cudaHelper.m_GRAVITY = this->m_GRAVITY;
	std::cout << "Gravity: ";
	if (m_GRAVITY) {
		std::cout << "ON" << std::endl;
		for (unsigned i = 0; i < m_numParticles; ++i) {
			m_particles[i].m_velocity[0] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f) * 2.0f;
			m_particles[i].m_velocity[1] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f) * 2.0f;
			m_particleVel[i * 2] = m_particles[i].m_velocity[0];
			m_particleVel[i * 2 + 1] = m_particles[i].m_velocity[1];
		}
	}
	else 
		std::cout << "OFF" << std::endl;
}


// Prints particle positions 
void ParticleSystem::printParticlePos(){
	for (unsigned i = 0; i < m_numParticles; ++i) {
		std::cout << "Position: " << i + 1 << " -> " << m_particles[i].m_position[0] << " : " << m_particles[i].m_position[1] << std::endl;
	}
}

// Prints particle velocities 
void ParticleSystem::printParticleVel() {
	for (unsigned i = 0; i < m_numParticles; ++i) {
		std::cout << "Velocity: " << i + 1 << " -> " << m_particles[i].m_velocity[0] << " : " << m_particles[i].m_velocity[1] << std::endl;
	}
}

