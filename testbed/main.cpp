#include "veekay/input.hpp"
#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

// #define USE_MATH_DEFINES
#include <cmath>
#include <map>

#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>
#include <lodepng.h>
#include <ranges>

#include "tiny_obj_loader.h"


enum class SamplerMode {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
    ClampToBorder
};

namespace {
    constexpr uint32_t MAX_MODELS = 1024;
    constexpr uint32_t MAX_LIGHTS = 4;


    struct Vertex {
        veekay::vec3 position;
        veekay::vec3 normal;
        veekay::vec2 uv;
    };

    struct LightColors {
        veekay::vec3 ambient;
        float _pad0;
        veekay::vec3 diffuse;
        float _pad1;
        veekay::vec3 specular;
        float _pad2;
    };

    struct DirectionalLight {
        veekay::vec3 direction;
        float intensity;
        LightColors colors;
    };

    struct PointLight {
        veekay::vec3 position = {};
        float _pad0;
        LightColors colors = {
            .ambient = {0.0f, 0.0f, 0.0f},
            .diffuse = {1.0f, 1.0f, 1.0f},
            .specular = {1.0f, 1.0f, 1.0f}
        };
        float linear = 0.09f;
        float quadratic = 0.032f;
        float _pad1;
        float _pad2;
    };

    struct AmbientLight {
        veekay::vec3 color;
        float intensity;
        veekay::vec3 specular_color;
        float shininess;
    };

    struct alignas(16) SceneUniforms {
        veekay::mat4 view_projection;
        veekay::vec3 camera_position;
        uint32_t num_point_lights = 0;
        uint32_t num_spot_lights = 0;
        uint32_t _pad_align[3];
        DirectionalLight directional_light;
        AmbientLight ambient_light;
    };

    struct ModelUniforms {
        veekay::mat4 model{};
        veekay::vec3 albedo_color{};
        float shininess = 10.0f;
        veekay::vec3 specular_color{};
        float _pad0{};
    };

    struct Mesh {
        veekay::graphics::Buffer *vertex_buffer = nullptr;
        veekay::graphics::Buffer *index_buffer = nullptr;
        uint32_t indices = 0;
    };

    class Transform {
    public:
        veekay::vec3 position = {};
        veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
        veekay::vec3 rotation = {};

        [[nodiscard]] veekay::mat4 matrix() const;
    };

    struct TextureAsset {
        veekay::graphics::Texture *texture = nullptr;
        std::string path;
    };

    struct Model {
        Mesh mesh;
        Transform transform;
        veekay::vec3 albedo_color = {1.0f, 1.0f, 1.0f};
        veekay::vec3 specular_color = {0.5f, 0.5f, 0.5f};
        float shininess = 10.0f;

        veekay::graphics::Texture *texture_ref = nullptr;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

        SamplerMode sampler_mode = SamplerMode::Repeat;
    };

    class Camera {
    public:
        constexpr static float DEFAULT_FOV = 60.0f;
        constexpr static float DEFAULT_NEAR_PLANE = 0.01f;
        constexpr static float DEFAULT_FAR_PLANE = 100.0f;

        veekay::vec3 position = {};
        veekay::vec3 rotation = {};

        float fov = DEFAULT_FOV;
        float near_plane = DEFAULT_NEAR_PLANE;
        float far_plane = DEFAULT_FAR_PLANE;

        [[nodiscard]] veekay::mat4 view() const;

        [[nodiscard]] veekay::mat4 view_projection(float aspect_ratio) const;

        static float toRadians(float degrees) {
            return degrees * static_cast<float>(M_PI) / 180.0f;
        }

        [[nodiscard]] veekay::vec3 getFront() const {
            const float pitch = rotation.x;
            const float yaw = rotation.y;

            const veekay::vec3 front = {
                std::cos(pitch) * std::sin(yaw),
                std::sin(pitch),
                -std::cos(pitch) * std::cos(yaw)
            };
            return veekay::vec3::normalized(front);
        }
    };

    struct SpotLight {
        PointLight point_light;
        veekay::vec3 direction;
        float cut_off = std::cos(Camera::toRadians(12.5f));
        float outer_cut_off = std::cos(Camera::toRadians(15.0f));
        float _pad1;
        float _pad2;
    };

    class Renderer {
        struct MeshPart {
            Mesh mesh;
            std::string material_name;
        };

        std::map<SamplerMode, VkSampler> samplers;

        VkShaderModule vertex_shader_module = VK_NULL_HANDLE;
        VkShaderModule fragment_shader_module = VK_NULL_HANDLE;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;

        veekay::graphics::Buffer *scene_uniforms_buffer = nullptr;
        veekay::graphics::Buffer *model_uniforms_buffer = nullptr;

        veekay::graphics::Buffer *point_lights_buffer = nullptr;
        veekay::graphics::Buffer *spot_lights_buffer = nullptr;

        Mesh plane_mesh;
        Mesh cube_mesh;

        veekay::graphics::Texture *missing_texture = nullptr;
        VkSampler missing_texture_sampler = VK_NULL_HANDLE;
        veekay::graphics::Texture *texture = nullptr;
        VkSampler texture_sampler = VK_NULL_HANDLE;
        std::vector<TextureAsset> loaded_textures;
        veekay::graphics::Texture *default_texture = nullptr;

        veekay::graphics::Texture *loadTexture(VkCommandBuffer cmd, const char *path);

        std::vector<MeshPart> loadObjParts(VkCommandBuffer cmd, const char *filename);

        static VkShaderModule loadShaderModule(const char *path);

        void createMeshes(VkCommandBuffer cmd);

        void createUniformsAndDescriptors(VkCommandBuffer cmd);

        void allocateDescriptorsForModels(VkCommandBuffer cmd);

        VkSampler createSampler(VkSamplerAddressMode addressMode);

    public:
        Renderer() = default;

        Renderer(const Renderer &) = delete;

        Renderer &operator=(const Renderer &) = delete;

        void initialize(VkCommandBuffer cmd);

        void shutdown() const;

        void update_uniforms(const Camera &camera, const std::vector<Model> &models, float aspect_ratio,
                             const std::vector<PointLight> &point_lights,
                             const std::vector<SpotLight> &spot_lights,
                             const DirectionalLight &dir_light,
                             const AmbientLight &ambient_light) const;

        void render(VkCommandBuffer cmd, VkFramebuffer framebuffer, const std::vector<Model> &models) const;

        [[nodiscard]] const Mesh &getPlaneMesh() const { return plane_mesh; }
        [[nodiscard]] const Mesh &getCubeMesh() const { return cube_mesh; }

        void createDefaultTexture(VkCommandBuffer cmd);
    };

    namespace Scene {
        Camera camera{
            .position = {0.0f, -0.5f, -3.0f}
        };
        std::vector<Model> models;
        Renderer renderer;

        DirectionalLight dir_light{
            .direction = {-0.338f, 0.761f, -0.316f},
            .intensity = 0.203f,
            .colors = {
                .ambient = {1.0f, 1.0f, 1.0f},
                .diffuse = {1.0f, 1.0f, 1.0f},
                .specular = {1.0f, 1.0f, 1.0f}
            }
        };

        AmbientLight ambient_light{
            .color = {0.0f, 0.0f, 0.0f},
            .intensity = 0.015f,
            .specular_color = {0.0f, 0.0f, 0.0f},
            .shininess = 8.0f,
        };
        std::vector<PointLight> point_lights = {
            {
                .position = {0.637f, -2.574f, -0.091f},
                .colors = {
                    .ambient = {0.0f, 0.0f, 0.0f},
                    .diffuse = {1.0f, 1.0f, 1.0f},
                    .specular = {1.0f, 0.0f, 0.0f}
                }
            },
        };
        std::vector<SpotLight> spot_lights = {
            {
                .point_light = {
                    .position = {0.584f, -6.344f, 0.045f},
                    .colors = {
                        .ambient = {0.0f, 0.0f, 0.0f},
                        .diffuse = {0.0f, 1.0f, 0.0f},
                        .specular = {0.0f, 1.0f, 0.0f}
                    }
                },
                .direction = {0.0f, 1.0f, 0.0f}
            }
        };
    }


    veekay::mat4 Transform::matrix() const {
        // 1. Масштабирование
        const auto s = veekay::mat4::scaling(scale);

        // 2. Поворот (Z-Y-X порядок, используя mat4::rotation из types.hpp)
        const auto rot_x = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, rotation.x);
        const auto rot_y = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, rotation.y);
        const auto rot_z = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, rotation.z);

        const auto r = rot_z * rot_y * rot_x;

        // 3. Смещение
        const auto t = veekay::mat4::translation(position);

        // Модельная матрица = T * R * S
        return t * r * s;
    }

    veekay::mat4 Camera::view() const {
        const veekay::vec3 front = getFront();

        const veekay::vec3 target = position + front;

        veekay::vec3 up = {0.0f, 1.0f, 0.0f};
        return veekay::mat4::lookAt(position, target, up);
    }

    veekay::mat4 Camera::view_projection(float aspect_ratio) const {
        auto projection = veekay::mat4::projection(
            fov, aspect_ratio, near_plane, far_plane);

        return view() * projection;
    }

    VkShaderModule Renderer::loadShaderModule(const char *path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) return VK_NULL_HANDLE;

        const size_t size = file.tellg();
        std::vector<uint32_t> buffer(size / sizeof(uint32_t));
        file.seekg(0);
        file.read(reinterpret_cast<char *>(buffer.data()), size);
        file.close();

        VkShaderModuleCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = size,
            .pCode = buffer.data(),
        };

        VkShaderModule result;
        if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
            return VK_NULL_HANDLE;
        }

        return result;
    }

    void Renderer::createMeshes(VkCommandBuffer cmd) { {
            std::vector<Vertex> vertices = {
                {{-5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
                {{5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {5.0f, 0.0f}},
                {{5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {5.0f, 5.0f}},
                {{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 5.0f}},
            };
            std::vector<uint32_t> indices = {0, 1, 2, 2, 3, 0};
            plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
                vertices.size() * sizeof(Vertex), vertices.data(),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            plane_mesh.index_buffer = new veekay::graphics::Buffer(
                indices.size() * sizeof(uint32_t), indices.data(),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            plane_mesh.indices = uint32_t(indices.size());
        } {
            std::vector<Vertex> vertices = {
                {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
                {{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
                {{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
                {{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},

                {{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
                {{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
                {{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
                {{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

                {{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
                {{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
                {{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
                {{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},

                {{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
                {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
                {{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
                {{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

                {{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
                {{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
                {{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
                {{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},

                {{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
                {{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
                {{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
                {{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
            };
            std::vector<uint32_t> indices = {
                0, 1, 2, 2, 3, 0,
                4, 5, 6, 6, 7, 4,
                8, 9, 10, 10, 11, 8,
                12, 13, 14, 14, 15, 12,
                16, 17, 18, 18, 19, 16,
                20, 21, 22, 22, 23, 20,
            };
            cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
                vertices.size() * sizeof(Vertex), vertices.data(),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            cube_mesh.index_buffer = new veekay::graphics::Buffer(
                indices.size() * sizeof(uint32_t), indices.data(),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            cube_mesh.indices = static_cast<uint32_t>(indices.size());
        }
    }

    void Renderer::createUniformsAndDescriptors(VkCommandBuffer cmd) {
        VkDevice &device = veekay::app.vk_device;

        scene_uniforms_buffer = new veekay::graphics::Buffer(
            sizeof(SceneUniforms), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        model_uniforms_buffer = new veekay::graphics::Buffer(
            MAX_MODELS * sizeof(ModelUniforms), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        point_lights_buffer = new veekay::graphics::Buffer(
            MAX_LIGHTS * sizeof(PointLight), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        spot_lights_buffer = new veekay::graphics::Buffer(
            MAX_LIGHTS * sizeof(SpotLight), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT); {
            samplers[SamplerMode::Repeat] = createSampler(VK_SAMPLER_ADDRESS_MODE_REPEAT);
            samplers[SamplerMode::MirroredRepeat] = createSampler(VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT);
            samplers[SamplerMode::ClampToEdge] = createSampler(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
            samplers[SamplerMode::ClampToBorder] = createSampler(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER);
        } {
            VkDescriptorPoolSize pools[] = {
                {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = MAX_MODELS},
                {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = MAX_MODELS},
                {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = MAX_MODELS},
                {.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = MAX_MODELS},
                {.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = MAX_MODELS}
            };
            VkDescriptorPoolCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = MAX_MODELS, .poolSizeCount = std::size(pools), .pPoolSizes = pools,
            };
            if (vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool) != VK_SUCCESS)
                throw std::runtime_error("Failed to create Vulkan descriptor pool.");
        } {
            VkDescriptorSetLayoutBinding bindings[] = {
                {
                    .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
                },
                {
                    .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
                },
                {
                    .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
                },
                {
                    .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
                },
                {
                    .binding = 4, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
                }
            };
            VkDescriptorSetLayoutCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = std::size(bindings), .pBindings = bindings,
            };
            if (vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout) != VK_SUCCESS)
                throw std::runtime_error("Failed to create Vulkan descriptor set layout.");
        }
    }

    void Renderer::allocateDescriptorsForModels(VkCommandBuffer cmd) {
        VkDevice &device = veekay::app.vk_device;

        for (auto &model: Scene::models) {
            VkDescriptorSetAllocateInfo alloc_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = descriptor_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &descriptor_set_layout,
            };
            // This will now work because descriptor_pool and layout exist
            if (vkAllocateDescriptorSets(device, &alloc_info, &model.descriptor_set) != VK_SUCCESS) {
                throw std::runtime_error("Failed to allocate descriptor set!");
            }

            veekay::graphics::Texture *tex = model.texture_ref ? model.texture_ref : default_texture;

            VkSampler selected_sampler = samplers[model.sampler_mode];

            VkDescriptorBufferInfo buffer_infos[] = {
                {.buffer = scene_uniforms_buffer->buffer, .offset = 0, .range = sizeof(SceneUniforms)},
                {.buffer = model_uniforms_buffer->buffer, .offset = 0, .range = sizeof(ModelUniforms)},
                {.buffer = point_lights_buffer->buffer, .offset = 0, .range = MAX_LIGHTS * sizeof(PointLight)},
                {.buffer = spot_lights_buffer->buffer, .offset = 0, .range = MAX_LIGHTS * sizeof(SpotLight)},
            };

            VkDescriptorImageInfo image_info{
                .sampler = selected_sampler,
                .imageView = tex->view,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            };

            VkWriteDescriptorSet writes[] = {
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = model.descriptor_set, .dstBinding = 0,
                    .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = &buffer_infos[0]
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = model.descriptor_set, .dstBinding = 1,
                    .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                    .pBufferInfo = &buffer_infos[1]
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = model.descriptor_set, .dstBinding = 2,
                    .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &buffer_infos[2]
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = model.descriptor_set, .dstBinding = 3,
                    .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &buffer_infos[3]
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = model.descriptor_set, .dstBinding = 4,
                    .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &image_info
                }
            };
            vkUpdateDescriptorSets(device, std::size(writes), writes, 0, nullptr);
        }
    }

    void Renderer::initialize(VkCommandBuffer cmd) {
        VkDevice &device = veekay::app.vk_device;

        vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
        if (!vertex_shader_module) throw std::runtime_error("Failed to load Vulkan vertex shader from file.");
        fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
        if (!fragment_shader_module) throw std::runtime_error("Failed to load Vulkan fragment shader from file.");

        createUniformsAndDescriptors(cmd);
        createDefaultTexture(cmd);

        auto *lenna = loadTexture(cmd, "./assets/lenna.png");
        auto *road = loadTexture(cmd, "./assets/road.png");

        auto *tex_body = loadTexture(cmd, "./assets/mcqueen/textures/body.png");
        auto *tex_eyes = loadTexture(cmd, "./assets/mcqueen/textures/eyes.png");
        auto *tex_tires = loadTexture(cmd, "./assets/mcqueen/textures/tires.png");


        auto carParts = loadObjParts(cmd, "./assets/mcqueen/LightingMcqueen.obj");

        createMeshes(cmd);

        VkPipelineShaderStageCreateInfo stage_infos[2];
        stage_infos[0] = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertex_shader_module, .pName = "main"
        };
        stage_infos[1] = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragment_shader_module, .pName = "main"
        };

        VkVertexInputBindingDescription buffer_binding{
            .binding = 0, .stride = sizeof(Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
        };
        VkVertexInputAttributeDescription attributes[] = {
            {
                .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, position)
            },
            {.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal)},
            {.location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(Vertex, uv)},
        };

        VkPipelineVertexInputStateCreateInfo input_state_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &buffer_binding, .vertexAttributeDescriptionCount = std::size(attributes),
            .pVertexAttributeDescriptions = attributes
        };
        VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
        };
        VkPipelineRasterizationStateCreateInfo raster_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
            .frontFace = VK_FRONT_FACE_CLOCKWISE, .lineWidth = 1.0f
        };
        VkPipelineMultisampleStateCreateInfo sample_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT, .sampleShadingEnable = false, .minSampleShading = 1.0f
        };

        VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic_state_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = std::size(dynamic_states),
            .pDynamicStates = dynamic_states
        };

        VkPipelineViewportStateCreateInfo viewport_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1
        };
        VkPipelineDepthStencilStateCreateInfo depth_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, .depthTestEnable = true,
            .depthWriteEnable = true, .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL
        };
        VkPipelineColorBlendAttachmentState attachment_info{
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                              VK_COLOR_COMPONENT_A_BIT
        };
        VkPipelineColorBlendStateCreateInfo blend_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .logicOpEnable = false,
            .logicOp = VK_LOGIC_OP_COPY, .attachmentCount = 1, .pAttachments = &attachment_info
        };

        VkPipelineLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
            .pSetLayouts = &descriptor_set_layout
        };
        if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan pipeline layout.");

        VkGraphicsPipelineCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2, .pStages = stage_infos,
            .pVertexInputState = &input_state_info, .pInputAssemblyState = &assembly_state_info,
            .pViewportState = &viewport_info,
            .pRasterizationState = &raster_info, .pMultisampleState = &sample_info,
            .pDepthStencilState = &depth_info,
            .pColorBlendState = &blend_info, .pDynamicState = &dynamic_state_info,
            .layout = pipeline_layout, .renderPass = veekay::app.vk_render_pass,
        };

        if (vkCreateGraphicsPipelines(device, nullptr, 1, &info, nullptr, &pipeline) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan pipeline.");

        for (const auto &part: carParts) {
            Model carModel;
            carModel.sampler_mode = SamplerMode::ClampToBorder;
            carModel.mesh = part.mesh;
            carModel.transform = Transform{.position = {0.0f, 0.0f, 0.0f}, .scale = {-0.7f, -0.7f, 0.7f}};
            carModel.shininess = 100.0f;

            if (part.material_name.find("Character_Eyes_TEX") != std::string::npos) {
                carModel.texture_ref = tex_eyes;
            } else if (part.material_name.find("Character_Mcqueen_Wheel_MAT") != std::string::npos) {
                carModel.texture_ref = tex_tires;
            } else {
                carModel.texture_ref = tex_body;
            }

            Scene::models.push_back(carModel);
        }


        Scene::models.emplace_back(Model{
            .mesh = getPlaneMesh(),
            .transform = Transform{},
            .albedo_color = veekay::vec3{0.7f, 0.7f, 0.7f},
            .specular_color = veekay::vec3{0.05f, 0.05f, 0.05f},
            .shininess = 32.0f,
            .texture_ref = road,
            .sampler_mode = SamplerMode::MirroredRepeat,
        });

        Scene::models.emplace_back(Model{
            .mesh = getCubeMesh(),
            .transform = Transform{.position = {-2.0f, -0.5f, -1.5f}},
            .albedo_color = veekay::vec3{1.0f, 0.0f, 0.0f},
            .specular_color = veekay::vec3{0.6f, 0.6f, 0.6f},
            .shininess = 64.0f,
            .texture_ref = lenna,
            .sampler_mode = SamplerMode::Repeat
        });

        Scene::models.emplace_back(Model{
            .mesh = getCubeMesh(),
            .transform = Transform{.position = {2.0f, -0.5f, -0.5f}},
            .albedo_color = veekay::vec3{0.0f, 1.0f, 0.0f},
            .specular_color = veekay::vec3{0.8f, 0.8f, 0.8f},
            .shininess = 32.0f,
            .texture_ref = lenna,
            .sampler_mode = SamplerMode::MirroredRepeat
        });

        allocateDescriptorsForModels(cmd);
    }

    void Renderer::shutdown() const {
        VkDevice &device = veekay::app.vk_device;

        vkDestroySampler(device, texture_sampler, nullptr);

        if (default_texture) delete default_texture;
        for (auto &asset: loaded_textures) {
            delete asset.texture;
        }

        for (const auto &sampler: samplers | std::views::values) {
            vkDestroySampler(device, sampler, nullptr);
        }

        delete missing_texture;

        for (const auto &model: Scene::models) {
            bool is_plane = (model.mesh.vertex_buffer == plane_mesh.vertex_buffer);
            bool is_cube = (model.mesh.vertex_buffer == cube_mesh.vertex_buffer);

            if (!is_plane && !is_cube) {
                delete model.mesh.index_buffer;
                delete model.mesh.vertex_buffer;
            }
        }

        delete cube_mesh.index_buffer;
        delete cube_mesh.vertex_buffer;
        delete plane_mesh.index_buffer;
        delete plane_mesh.vertex_buffer;

        delete spot_lights_buffer;
        delete point_lights_buffer;

        delete model_uniforms_buffer;
        delete scene_uniforms_buffer;

        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyShaderModule(device, fragment_shader_module, nullptr);
        vkDestroyShaderModule(device, vertex_shader_module, nullptr);
    }

    veekay::graphics::Texture *Renderer::loadTexture(VkCommandBuffer cmd, const char *path) {
        for (auto &asset: loaded_textures) {
            if (asset.path == path) return asset.texture;
        }

        std::vector<unsigned char> pixels;
        unsigned int width, height;

        if (unsigned error = lodepng::decode(pixels, width, height, path)) {
            std::cerr << "Texture load error [" << path << "]: " << lodepng_error_text(error) << "\n";
            return default_texture;
        }

        auto *new_tex = new veekay::graphics::Texture(
            cmd, width, height, VK_FORMAT_R8G8B8A8_UNORM, pixels.data()
        );

        loaded_textures.push_back({new_tex, path});
        return new_tex;
    }

    std::vector<Renderer::MeshPart> Renderer::loadObjParts(VkCommandBuffer cmd, const char *filename) {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        // Важно: base_dir должен указывать на папку, где лежит .mtl файл (обычно там же где obj)
        std::string base_dir = std::string(filename).substr(0, std::string(filename).find_last_of("/\\") + 1);

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename, base_dir.c_str())) {
            throw std::runtime_error("LoadObj error: " + warn + err);
        }

        std::vector<MeshPart> parts;

        // Группируем фигуры по индексу материала
        // Key: material_index, Value: list of indices
        std::map<int, std::vector<uint32_t> > material_to_indices;
        std::map<int, std::vector<Vertex> > material_to_vertices;

        // В tinyobjloader фигуры (shapes) могут делить материалы.
        // Нам нужно "слить" все треугольники с одинаковым материалом в один меш.
        for (const auto &shape: shapes) {
            for (size_t f = 0; f < shape.mesh.indices.size() / 3; f++) {
                int mat_id = shape.mesh.material_ids[f];

                // Если материал не назначен, используем -1 (дефолтный)
                if (mat_id < 0) mat_id = -1;

                // Обрабатываем 3 вершины треугольника
                for (int v = 0; v < 3; v++) {
                    tinyobj::index_t idx = shape.mesh.indices[3 * f + v];

                    Vertex vertex{};

                    // Позиция
                    vertex.position = {
                        attrib.vertices[3 * idx.vertex_index + 0],
                        attrib.vertices[3 * idx.vertex_index + 1],
                        attrib.vertices[3 * idx.vertex_index + 2]
                    };

                    // Нормаль
                    if (idx.normal_index >= 0) {
                        vertex.normal = {
                            attrib.normals[3 * idx.normal_index + 0],
                            attrib.normals[3 * idx.normal_index + 1],
                            attrib.normals[3 * idx.normal_index + 2]
                        };
                    }

                    // UV
                    if (idx.texcoord_index >= 0) {
                        vertex.uv = {
                            attrib.texcoords[2 * idx.texcoord_index + 0],
                            1.0f - attrib.texcoords[2 * idx.texcoord_index + 1]
                        };
                    }

                    material_to_vertices[mat_id].push_back(vertex);
                    material_to_indices[mat_id].push_back(material_to_indices[mat_id].size());
                }
            }
        }

        // Теперь создаем Mesh для каждого материала
        for (auto &[mat_id, verts]: material_to_vertices) {
            MeshPart part;

            // Пытаемся достать имя материала
            if (mat_id >= 0 && mat_id < materials.size()) {
                part.material_name = materials[mat_id].name;
            } else {
                part.material_name = "default";
            }

            part.mesh.vertex_buffer = new veekay::graphics::Buffer(
                verts.size() * sizeof(Vertex), verts.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
            );

            auto &indices = material_to_indices[mat_id];
            part.mesh.index_buffer = new veekay::graphics::Buffer(
                indices.size() * sizeof(uint32_t), indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT
            );
            part.mesh.indices = static_cast<uint32_t>(indices.size());

            parts.push_back(part);
        }

        return parts;
    }

    void Renderer::createDefaultTexture(VkCommandBuffer cmd) {
        uint32_t pixels[] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF}; // Pure White
        default_texture = new veekay::graphics::Texture(cmd, 2, 2, VK_FORMAT_B8G8R8A8_UNORM, pixels);
    }

    VkSampler Renderer::createSampler(VkSamplerAddressMode addressMode) {
        VkSamplerCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU = addressMode,
            .addressModeV = addressMode,
            .addressModeW = addressMode,

            .anisotropyEnable = VK_TRUE,
            .maxAnisotropy = 16.0f,
            .minLod = 0.0f,
            .maxLod = VK_LOD_CLAMP_NONE,
        };

        VkSampler sampler;
        if (vkCreateSampler(veekay::app.vk_device, &info, nullptr, &sampler) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create sampler");
        }
        return sampler;
    }

    void Renderer::update_uniforms(const Camera &camera, const std::vector<Model> &models, float aspect_ratio,
                                   const std::vector<PointLight> &point_lights,
                                   const std::vector<SpotLight> &spot_lights,
                                   const DirectionalLight &dir_light,
                                   const AmbientLight &ambient_light) const {
        SceneUniforms scene_uniforms{
            .view_projection = camera.view_projection(aspect_ratio),
            .camera_position = camera.position,
            .num_point_lights = static_cast<uint32_t>(point_lights.size()),
            .num_spot_lights = static_cast<uint32_t>(spot_lights.size()),
            ._pad_align = {0, 0},
            .directional_light = dir_light,
            .ambient_light = ambient_light
        };

        std::vector<ModelUniforms> model_uniforms(models.size());
        for (size_t i = 0; i < models.size(); ++i) {
            const Model &model = models[i];
            ModelUniforms &uniforms = model_uniforms[i];

            uniforms.model = model.transform.matrix();
            uniforms.albedo_color = model.albedo_color;
            uniforms.specular_color = model.specular_color;
            uniforms.shininess = model.shininess;
        }

        if (scene_uniforms_buffer->mapped_region)
            *static_cast<SceneUniforms *>(scene_uniforms_buffer->mapped_region) = scene_uniforms;

        if (model_uniforms_buffer->mapped_region)
            std::copy(model_uniforms.begin(),
                      model_uniforms.end(),
                      static_cast<ModelUniforms *>(model_uniforms_buffer->mapped_region));

        if (point_lights_buffer->mapped_region)
            std::copy(point_lights.begin(),
                      point_lights.end(),
                      static_cast<PointLight *>(point_lights_buffer->mapped_region));

        if (spot_lights_buffer->mapped_region)
            std::copy(spot_lights.begin(),
                      spot_lights.end(),
                      static_cast<SpotLight *>(spot_lights_buffer->mapped_region));
    }


    void Renderer::render(VkCommandBuffer cmd, VkFramebuffer framebuffer, const std::vector<Model> &models) const {
        vkResetCommandBuffer(cmd, 0); {
            VkCommandBufferBeginInfo info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };
            vkBeginCommandBuffer(cmd, &info);
        } {
            VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
            VkClearValue clear_depth{.depthStencil = {1.0f, 0}};
            VkClearValue clear_values[] = {clear_color, clear_depth};

            VkRenderPassBeginInfo info{
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .renderPass = veekay::app.vk_render_pass,
                .framebuffer = framebuffer,
                .renderArea = {.extent = {veekay::app.window_width, veekay::app.window_height}},
                .clearValueCount = std::size(clear_values),
                .pClearValues = clear_values,
            };
            vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        VkViewport viewport{
            .x = 0.0f, .y = 0.0f, .width = static_cast<float>(veekay::app.window_width),
            .height = static_cast<float>(veekay::app.window_height),
            .minDepth = 0.0f, .maxDepth = 1.0f,
        };
        VkRect2D scissor{
            .offset = {0, 0}, .extent = {veekay::app.window_width, veekay::app.window_height},
        };
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);


        VkDeviceSize zero_offset = 0;
        VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
        VkBuffer current_index_buffer = VK_NULL_HANDLE;

        for (size_t i = 0, n = models.size(); i < n; ++i) {
            const Model &model = models[i];
            const Mesh &mesh = model.mesh;

            if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
                current_vertex_buffer = mesh.vertex_buffer->buffer;
                vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
            }

            if (current_index_buffer != mesh.index_buffer->buffer) {
                current_index_buffer = mesh.index_buffer->buffer;
                vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
            }

            uint32_t offset = i * sizeof(ModelUniforms);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                    0, 1, &model.descriptor_set, 1, &offset);

            vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
    }


    void initialize(VkCommandBuffer cmd) {
        Scene::renderer.initialize(cmd);
    }

    void shutdown() {
        Scene::renderer.shutdown();
    }

    void update(double time) {
        ImGui::Begin("Controls:");
        ImGui::Text("Camera Position: (%.2f, %.2f, %.2f)", Scene::camera.position.x, Scene::camera.position.y,
                    Scene::camera.position.z);
        ImGui::Separator();
        if (ImGui::CollapsingHeader("Directional Light")) {
            ImGui::SliderFloat3("Dir Direction", &Scene::dir_light.direction.x, -1.0f, 1.0f);

            ImGui::SliderFloat("Intensity", &Scene::dir_light.intensity, 0.0f, 10.0f);

            ImGui::ColorEdit3("Dir Ambient", &Scene::dir_light.colors.ambient.x);
            ImGui::ColorEdit3("Dir Diffuse", &Scene::dir_light.colors.diffuse.x);
            ImGui::ColorEdit3("Dir Specular", &Scene::dir_light.colors.specular.x);
        }

        if (ImGui::CollapsingHeader("Ambient Light")) {
            ImGui::ColorEdit3("Ambient Color", &Scene::ambient_light.color.x);
            ImGui::SliderFloat("Ambient Intensity", &Scene::ambient_light.intensity, 0.0f, 5.0f);
            ImGui::ColorEdit3("Ambient Specular Color", &Scene::ambient_light.specular_color.x);
            ImGui::SliderFloat("Ambient Shininess", &Scene::ambient_light.shininess, 1.0f, 256.0f);
        }

        if (ImGui::CollapsingHeader("Point Light 1")) {
            PointLight &light = Scene::point_lights[0];
            ImGui::SliderFloat3("Pos Position", &light.position.x, -10.0f, 10.0f);
            ImGui::ColorEdit3("Pos Diffuse", &light.colors.diffuse.x);
            ImGui::ColorEdit3("Pos Specular", &light.colors.specular.x);
            ImGui::SliderFloat("Pos Linear Atten", &light.linear, 0.001f, 0.5f);
            ImGui::SliderFloat("Pos Quad Atten", &light.quadratic, 0.0001f, 0.1f);
        }

        if (ImGui::CollapsingHeader("Spot Light 1")) {
            SpotLight &light = Scene::spot_lights[0];
            ImGui::SliderFloat3("Spot Position", &light.point_light.position.x, -10.0f, 10.0f);
            ImGui::SliderFloat3("Spot Direction", &light.direction.x, -1.0f, 1.0f);
            light.direction = veekay::vec3::normalized(light.direction);
            ImGui::ColorEdit3("Spot Diffuse", &light.point_light.colors.diffuse.x);
            ImGui::ColorEdit3("Spot Specular", &light.point_light.colors.specular.x);

            float cutoff_deg = std::acos(light.cut_off) * 180.0f / static_cast<float>(M_PI);
            float outer_cutoff_deg = std::acos(light.outer_cut_off) * 180.0f / static_cast<float>(M_PI);

            ImGui::SliderFloat("Spot Cutoff (deg)", &cutoff_deg, 1.0f, 45.0f);
            ImGui::SliderFloat("Spot Outer Cutoff (deg)", &outer_cutoff_deg, 1.0f, 45.0f);

            if (outer_cutoff_deg < cutoff_deg) outer_cutoff_deg = cutoff_deg + 1.0f;

            light.cut_off = std::cos(cutoff_deg * static_cast<float>(M_PI) / 180.0f);
            light.outer_cut_off = std::cos(outer_cutoff_deg * static_cast<float>(M_PI) / 180.0f);

            ImGui::SliderFloat("Spot Linear Atten", &light.point_light.linear, 0.001f, 0.5f);
            ImGui::SliderFloat("Spot Quad Atten", &light.point_light.quadratic, 0.0001f, 0.1f);
        }

        ImGui::End();

        if (!ImGui::IsWindowHovered()) {
            using namespace veekay::input;
            Camera &camera = Scene::camera;

            constexpr float move_speed = 0.1f;

            if (mouse::isButtonDown(mouse::Button::left)) {
                constexpr float sensitivity = 0.003f;

                auto move_delta = mouse::cursorDelta();

                camera.rotation.y += move_delta[0] * sensitivity;
                camera.rotation.x += move_delta[1] * sensitivity;
            }
            const veekay::vec3 front = camera.getFront();
            veekay::vec3 up = {0.0f, 1.0f, 0.0f};

            const veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(front, up));
            up = veekay::vec3::normalized(veekay::vec3::cross(right, front));


            if (keyboard::isKeyDown(keyboard::Key::w)) camera.position -= front * move_speed;
            if (keyboard::isKeyDown(keyboard::Key::s)) camera.position += front * move_speed;
            if (keyboard::isKeyDown(keyboard::Key::d)) camera.position += right * move_speed;
            if (keyboard::isKeyDown(keyboard::Key::a)) camera.position -= right * move_speed;

            if (keyboard::isKeyDown(keyboard::Key::q)) camera.position -= up * move_speed;
            if (keyboard::isKeyDown(keyboard::Key::z)) camera.position += up * move_speed;
        }

        float aspect_ratio = static_cast<float>(veekay::app.window_width) / static_cast<float>(veekay::app.
                                 window_height);
        Scene::renderer.update_uniforms(Scene::camera, Scene::models, aspect_ratio,
                                        Scene::point_lights, Scene::spot_lights,
                                        Scene::dir_light, Scene::ambient_light);
    }

    void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
        Scene::renderer.render(cmd, framebuffer, Scene::models);
    }
} // namespace

int main() {
    return veekay::run({
        .init = initialize,
        .shutdown = shutdown,
        .update = update,
        .render = render,
    });
}