#version 460 core

out vec4 FragColor; 

uniform float Radius;
uniform vec2 Resolution;
uniform int NumParticles;
//uniform vec2 ParticleCoords[510];
//uniform vec3 ParticleColors[510];

layout(std430, binding = 0) buffer ParticleCoordsBuffer {
	vec2 ParticleCoords[];
};
layout(std430, binding = 1) buffer ParticleColorsBuffer {
	vec3 ParticleColors[];
};

void main() {
	vec2 fragCoord = (gl_FragCoord.xy / Resolution) * 2.0f - 1;
	for (int i = 0; i < NumParticles; ++i) { 
		float dist = distance(fragCoord, vec2(ParticleCoords[i][0], ParticleCoords[i][1]));
		
		if (dist <= Radius) {
			FragColor = vec4(ParticleColors[i], 1.0f);
			break;
		}
	}
}