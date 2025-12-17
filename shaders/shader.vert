#version 450

layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform PushConstants {
	mat4 mvp;   // Было: mat4 projection; mat4 transform;
	vec3 color;
} push;

layout(location = 0) out vec3 fragColor;

void main() {
	// Просто умножаем MVP на позицию
	gl_Position = push.mvp * vec4(inPosition, 1.0);

	fragColor = push.color;
}