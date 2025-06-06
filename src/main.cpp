#include <iostream>
#include <glad/glad.h> 
#include <iomanip>
#include "windows.h"
#include <GLFW/glfw3.h>
#include "../include/ParticleSystem.h"
#include "../include/Shader.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

const unsigned int WINDOW_SIZE = 800;
const unsigned int NUM_PARTICLES = 1000;
const float RADIUS = 0.005f;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window, ParticleSystem& particleSystem);
void calculateFPS(unsigned& runningFrameCount, long long& totalFrames);

int main() {
	std::cout << std::fixed << std::setprecision(10); 

	// GLFW initialization
	glfwInit(); 
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Window creation
	GLFWwindow* window = glfwCreateWindow(WINDOW_SIZE, WINDOW_SIZE, "Particle Simulator", NULL, NULL);

	if (window == nullptr) {
		std::cout << "GLFW Window creation failed" << std::endl;
		glfwTerminate();
		return -1;
	}

	// Turn off vSync manually
	glfwSwapInterval(0);

	// Context 
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	// Glad initialization
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	int maxUniformComponents;
	glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_COMPONENTS, &maxUniformComponents);
	std::cout << "Max fragment uniform components: " << maxUniformComponents << std::endl;

	const GLubyte* version = glGetString(GL_VERSION);
	std::cout << "OpenGL Versions:  " << version << std::endl;

	// Shader Program initialization
	Shader shaderProgram("shaders/vert.glsl", "shaders/frag.glsl");

	// Regular vs testing constructor
	ParticleSystem particleSystem(NUM_PARTICLES, WINDOW_SIZE, RADIUS, true);

	float vertices[] = {
		-1.0f, -1.0f, 0.0f,		// bottom left  
		 1.0f, -1.0f, 0.0f,		// bottom right 
		 -1.0f,  1.0f, 0.0f,	// top left
		 1.0f, 1.0f, 0.0f		// top right
	};

	// EBO vertices
	unsigned int indices[] = {	
		1, 0, 2,
		2, 3, 1
	};

	// Vertex Array initialization and binding
	unsigned int VAO;
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	// Vertex Buffer initialization and binding
	unsigned int VBO;
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	// Indices Array initialization and binding 
	unsigned int EBO;
	glGenBuffers(1, &EBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
	
	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	// VBO
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Activate the shader program
	shaderProgram.use();

	// Handle uniforms and other information
	int radiusLoc = glGetUniformLocation(shaderProgram.shaderProgramID, "Radius");
	glUniform1f(radiusLoc, RADIUS);
	std::cout << "UNIFORM LOCATION::Radius -> " << radiusLoc << std::endl;

	int resolutionLoc = glGetUniformLocation(shaderProgram.shaderProgramID, "Resolution");
	glUniform2f(resolutionLoc, (float)WINDOW_SIZE, (float)WINDOW_SIZE);
	std::cout << "UNIFORM LOCATION::Resolution -> " << resolutionLoc << std::endl;

	int numParticlesLoc = glGetUniformLocation(shaderProgram.shaderProgramID, "NumParticles");
	glUniform1i(numParticlesLoc, NUM_PARTICLES);
	std::cout << "UNIFORM LOCATION::NumParticles -> " << numParticlesLoc << std::endl;

	// SSBO for particles
	GLuint SSBO_ParticlePos;
	glGenBuffers(1, &SSBO_ParticlePos);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO_ParticlePos);
	glBufferData(GL_SHADER_STORAGE_BUFFER, NUM_PARTICLES * 2 * sizeof(float), particleSystem.getParticlePos(), GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, SSBO_ParticlePos);

	// SSBO for particle colors
	GLuint SSBO_ParticleColor;
	glGenBuffers(1, &SSBO_ParticleColor);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO_ParticleColor);
	glBufferData(GL_SHADER_STORAGE_BUFFER, NUM_PARTICLES * 4 * sizeof(float), particleSystem.getParticleColor(), GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, SSBO_ParticleColor);

	//cudaGraphicsResource* cudaSSBO;
	//// Cuda write flag COME BACK
	//cudaGraphicsGLRegisterBuffer(&cudaSSBO, SSBO_ParticlePos, cudaGraphicsMapFlagsWriteDiscard);

	//cudaGraphicsMapResources(1, &cudaSSBO);
	//float* dptr;
	//size_t num_bytes;
	//cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cudaSSBO);

	//// Kernel Call

	//// Unmap SSBO so OpenGL can use it
	//cudaGraphicsUnmapResources(1, &cudaSSBO);

	// Enable depth testing
	glEnable(GL_DEPTH_TEST);

	GLenum err;

	auto prevTime = GetTickCount64();
	auto currTime = GetTickCount64();

	unsigned runningFrameCount = 0;
	long long totalFrames = 0;

	while (!glfwWindowShouldClose(window)) {
		processInput(window, particleSystem);

		// Set color to clear color buffer	
		glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
		// Clear color buffer with preset color
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Particle system logic
		currTime = GetTickCount64();
		if (particleSystem.isRunning()) {
			float deltaTime = (currTime - prevTime) / 1000.0f;
			if (deltaTime) {
				particleSystem.simulate(deltaTime);
				
				// SSBO buffer binding
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO_ParticlePos);
				float* ptr = (float*)glMapBufferRange(
					GL_SHADER_STORAGE_BUFFER,
					0,
					NUM_PARTICLES * 2 * sizeof(float),
					GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT
				);
				if (ptr) {
					memcpy(ptr, particleSystem.getParticlePos(), NUM_PARTICLES * 2 * sizeof(float));
					glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
				}
				else
					std::cerr << "Buffer mapping failed!" << std::endl;
			}
		}
		prevTime = currTime;

		// Used to draw from EBO
		glDrawElements(GL_TRIANGLES, sizeof(indices) / sizeof(unsigned), GL_UNSIGNED_INT, 0);

		calculateFPS(runningFrameCount, totalFrames);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	while ((err = glGetError()) != GL_NO_ERROR) 
		std::cout << "OpenGL Error: " << err << std::endl;
	
	glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback([](GLenum, GLenum, GLuint, GLenum severity, GLsizei, const GLchar* message, const void*) {
		std::cerr << "GL Debug: " << message << std::endl;
		}, nullptr);

	// Cleanup buffers and shaders
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &EBO);
	shaderProgram.clean();

	// Clearing all allocated GLFW resources
	glfwTerminate();
	return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}

void processInput(GLFWwindow* window, ParticleSystem& particleSystem) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
		particleSystem.toggleRunning();
	if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS)
		particleSystem.toggleGravity();
}

void calculateFPS(unsigned& runningFrameCount, long long& totalFrames) {
	static int frames = 0;
	static float lastTime = 0.0f;
	auto currentTime = GetTickCount64() * 0.001f;

	frames++;
	if (currentTime - lastTime >= 1.0f) {
		totalFrames += frames;
		runningFrameCount++;
		std::cout << "FPS: " << frames << std::endl;
		std::cout << "Average FPS: " << totalFrames / runningFrameCount - 1 << std::endl;
		lastTime = currentTime;
		frames = 0;
	}
}