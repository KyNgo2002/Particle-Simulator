#version 460 core
//precision highp float;

out vec4 FragColor; 

uniform float Radius;
uniform vec2 Resolution;
uniform vec2 Origin;
uniform int NumParticles;
uniform vec2 ParticleCoords[1000];
uniform vec3 ParticleColors[1000];

void main() {
	vec2 fragCoord = (gl_FragCoord.xy / Resolution) * 2.0f - 1;
	
	for (int i = 0; i < NumParticles; ++i) { 
		float dist = distance(fragCoord, vec2(ParticleCoords[i][0], ParticleCoords[i][1]));
		if (dist <= Radius) {
			FragColor = vec4(ParticleColors[i].x, ParticleColors[i].y, ParticleColors[i].z, 1.0f);
		}
	}

}