#version 450

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;

layout (location = 0) out vec3 f_position; // world-space position
layout (location = 1) out vec3 f_normal;   // world-space normal (normalized)
layout (location = 2) out vec2 f_uv;

layout(std140, binding = 0) uniform SceneUniforms {
    mat4 view_projection;

    vec3 camera_position;
    uint num_point_lights;

    uint num_spot_lights;
    uint _pad_align0;
    uint _pad_align1;

    vec3 dir_direction;
    float dir_intensity;
    vec3 dir_ambient;
    float _pad_dir0;
    vec3 dir_diffuse;
    float _pad_dir1;
    vec3 dir_specular;
    float _pad_dir2;

    vec3 amb_color;
    float amb_intensity;
    vec3 amb_specular_color;
    float amb_shininess;
} scene;

layout(std140, binding = 1) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float shininess;
    vec3 specular_color;
    float _pad0;
} material;

void main() {
    vec4 world_pos = material.model * vec4(in_position, 1.0);
    f_position = world_pos.xyz;

    mat3 normal_matrix = transpose(inverse(mat3(material.model)));
    f_normal = normalize(normal_matrix * in_normal);

    f_uv = in_uv;

    gl_Position = scene.view_projection * world_pos;
}
