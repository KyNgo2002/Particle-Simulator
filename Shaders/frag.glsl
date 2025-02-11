#version 460 core

out vec4 FragColor; 

uniform float Radius;
uniform vec2 Resolution;
uniform int NumParticles;

layout(std430, binding = 0)  buffer ParticleCoordsBuffer {
	vec2 ParticleCoords[];
};
layout(std430, binding = 1) readonly buffer ParticleColorsBuffer {
	vec4 ParticleColors[];
};

void main() {
	vec2 fragCoord = (gl_FragCoord.xy / Resolution) * 2.0f - 1;
	for (int i = 0; i < NumParticles; i++) { 
		float dist = distance(fragCoord, vec2(ParticleCoords[i][0], ParticleCoords[i][1]));
		
		if (dist <= Radius) {
			FragColor = ParticleColors[i];
		}
	}
}