#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

// Если вы берете цвет из push constants напрямую во фрагментном:
layout(push_constant) uniform PushConstants {
	mat4 mvp;
	vec3 color;
} push;

void main() {
	outColor = vec4(push.color, 1.0);
}