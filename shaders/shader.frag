#version 450

layout(location = 0) in vec3 f_position;
layout(location = 1) in vec3 f_normal;
layout(location = 2) in vec2 f_uv;

layout(location = 0) out vec4 out_color;

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

struct SSBO_LightColors {
    vec3 ambient; float _pad0;
    vec3 diffuse; float _pad1;
    vec3 specular; float _pad2;
};

struct SSBO_PointLight {
    vec3 position; float _pad0;
    SSBO_LightColors colors;
    float linear;
    float quadratic;
    float _pad1;
    float _pad2;
};

layout(std430, binding = 2) readonly buffer PointLights {
    SSBO_PointLight lights[];
} pointLights;

struct SSBO_SpotLight {
    vec3 position; float _pad0;
    SSBO_LightColors colors;
    float linear;
    float quadratic;
    float _pad1;
    float _pad2;

    vec3 direction;
    float cut_off;
    float outer_cut_off;
    float _pad3;
    float _pad4;
};

layout(std430, binding = 3) readonly buffer SpotLights {
    SSBO_SpotLight spots[];
} spotLights;

vec3 calc_ambient_light(vec3 albedo, vec3 specular_color) {
    vec3 ambient = albedo * scene.amb_color * scene.amb_intensity;

    vec3 specular = specular_color * scene.amb_specular_color * scene.amb_intensity;

    return ambient + specular;
}

layout (binding = 4) uniform sampler2D albedo_texture;

vec3 calc_dir_light(vec3 N, vec3 V, vec3 albedo, vec3 specular_color, float shininess) {
    vec3 L = normalize(-scene.dir_direction);
    float NdotL = max(dot(N, L), 0.0);

    vec3 diffuse = scene.dir_diffuse * NdotL;

    vec3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);

    float spec_intensity = (NdotL > 0.0) ? pow(NdotH, shininess) : 0.0;
    vec3 specular = scene.dir_specular * spec_intensity;

    vec3 result = (albedo * diffuse + specular_color * specular);

    return result * scene.dir_intensity;
}

vec3 calc_point_light(SSBO_PointLight light, vec3 N, vec3 V, vec3 P, vec3 albedo, vec3 specular_col, float shininess) {
    vec3 L = light.position - P;
    float dist = length(L);
    L = normalize(L);

    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = light.colors.diffuse * NdotL;

    vec3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    vec3 specular = light.colors.specular * pow(NdotH, shininess);

    float attenuation = 1.0 / (1.0 + light.linear * dist + light.quadratic * dist * dist);

    vec3 ambient = light.colors.ambient;

    return (ambient + albedo * diffuse + specular_col * specular) * attenuation;
}

vec3 calc_spot_light(SSBO_SpotLight light, vec3 N, vec3 V, vec3 P, vec3 albedo, vec3 specular_col, float shininess) {
    vec3 L = light.position - P;
    float dist = length(L);
    L = normalize(L);

    vec3 spotDir = normalize(-light.direction);
    float theta = dot(L, spotDir);

    float epsilon = light.cut_off - light.outer_cut_off;
    float intensity = clamp((theta - light.outer_cut_off) / max(epsilon, 1e-6), 0.0, 1.0);

    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = light.colors.diffuse * NdotL;

    vec3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    vec3 specular = light.colors.specular * pow(NdotH, shininess);


    float attenuation = 1.0 / (1.0 + light.linear * dist + light.quadratic * dist * dist);

    vec3 ambient = light.colors.ambient;

    return (ambient + albedo * diffuse + specular_col * specular) * attenuation * intensity;
}

void main() {
    vec3 N = normalize(f_normal);
    vec3 P = f_position;
    vec3 V = normalize(scene.camera_position - P);

    vec4 texel = texture(albedo_texture, f_uv);

    vec3 albedo = material.albedo_color * texel.rgb;
    vec3 specular_col = material.specular_color;
    float shininess = material.shininess;

    vec3 color = vec3(0.0);

    color += calc_ambient_light(albedo, specular_col);

    color += calc_dir_light(N, V, material.albedo_color, material.specular_color, material.shininess);

    for (uint i = 0u; i < scene.num_point_lights; ++i) {
        color += calc_point_light(pointLights.lights[i], N, V, P, albedo, specular_col, shininess);
    }

    for (uint i = 0u; i < scene.num_spot_lights; ++i) {
        color += calc_spot_light(spotLights.spots[i], N, V, P, albedo, specular_col, shininess);
    }

    out_color = vec4(color, 1.0);
}
