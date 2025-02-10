#version 460 core

out vec4 FragColor; 

uniform float Radius;
uniform vec2 Resolution;
uniform int NumParticles;
uniform vec2 ParticleCoords[510];
uniform vec3 ParticleColors[510];

layout(std430, binding = 0) buffer ParticleColorsBuffer {
	vec3 ParticleColorss[];
};

void main() {
	vec2 fragCoord = (gl_FragCoord.xy / Resolution) * 2.0f - 1;
	
	for (int i = 0; i < NumParticles; ++i) { 
		vec3 particleColor = particleColorss[i];
		float dist = distance(fragCoord, vec2(ParticleCoords[i][0], ParticleCoords[i][1]));
		
		/*if (dist <= Radius) {
			FragColor = vec4(ParticleColors[i].x, ParticleColors[i].y, ParticleColors[i].z, 1.0f);
			break;
		}*/

		if (dist <= Radius) {
			FragColor = vec4(particleColor[i].x, particleColor[i].y, particleColor[i].z, 1.0f);
			break;
		}
	}
}