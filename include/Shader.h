#pragma once

#include <glad/glad.h>
#include <fstream>


class Shader {
public:
	// Shader Program ID
	unsigned int shaderProgramID;

	Shader(const char* vertexSource, const char* fragSource, const char* computeSource);
	void setUniform();
	void clean();
	void use();
};
