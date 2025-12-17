#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <string>

#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {
    struct Matrix {
        float m[4][4];
    };

    struct Vector {
        float x, y, z;
    };

    struct Vertex {
        Vector position;
    };

    struct ShaderConstants {
        Matrix mvp;
        Vector color;
    };

    struct VulkanBuffer {
        VkBuffer buffer;
        VkDeviceMemory memory;
    };

    struct Mesh {
        VulkanBuffer vertex_buffer;
        VulkanBuffer index_buffer;
        uint32_t index_count;
    };

    struct SceneObject {
        std::string name;
        Vector position;
        Vector color;
        Mesh *mesh_ptr;
    };

    constexpr float camera_fov = 60.0f;
    constexpr float camera_near_plane = 0.01f;
    constexpr float camera_far_plane = 100.0f;

    VkShaderModule vertex_shader_module;
    VkShaderModule fragment_shader_module;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;

    Mesh cube_mesh;
    Mesh pyramid_mesh;
    Mesh sphere_mesh;

    std::vector<SceneObject> scene_objects;

    float global_rotation_speed = 0.5f;
    float current_time = 0.0f;
    bool is_animating = true;

    Matrix identity() {
        Matrix result{};
        result.m[0][0] = 1.0f;
        result.m[1][1] = 1.0f;
        result.m[2][2] = 1.0f;
        result.m[3][3] = 1.0f;
        return result;
    }

    Matrix perspective(float fov, float aspect_ratio, float near_z, float far_z) {
        Matrix result{};
        const float radians = fov * (float) M_PI / 180.0f;
        const float cot = 1.0f / tanf(radians / 2.0f);
        result.m[0][0] = cot / aspect_ratio;
        result.m[1][1] = -cot;
        result.m[2][2] = far_z / (near_z - far_z);
        result.m[3][2] = -(far_z * near_z) / (far_z - near_z);
        result.m[2][3] = -1.0f;
        return result;
    }

    Matrix translation(Vector vector) {
        Matrix result = identity();
        result.m[3][0] = vector.x;
        result.m[3][1] = vector.y;
        result.m[3][2] = vector.z;
        return result;
    }

    Matrix rotation_y(float angle) {
        Matrix result = identity();
        float c = cosf(angle);
        float s = sinf(angle);
        result.m[0][0] = c;
        result.m[0][2] = -s;
        result.m[2][0] = s;
        result.m[2][2] = c;
        return result;
    }

    Matrix multiply(const Matrix &a, const Matrix &b) {
        Matrix result{};
        for (int j = 0; j < 4; j++) {
            for (int i = 0; i < 4; i++) {
                float sum = 0.0f;
                for (int k = 0; k < 4; k++) {
                    sum += a.m[j][k] * b.m[k][i];
                }
                result.m[j][i] = sum;
            }
        }
        return result;
    }

    VkShaderModule loadShaderModule(const char *path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) return nullptr;
        size_t size = file.tellg();
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
        vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result);
        return result;
    }

    VulkanBuffer createBuffer(size_t size, void *data, VkBufferUsageFlags usage) {
        VkDevice &device = veekay::app.vk_device;
        VkPhysicalDevice &physical_device = veekay::app.vk_physical_device;
        VulkanBuffer result{};

        VkBufferCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        vkCreateBuffer(device, &info, nullptr, &result.buffer);

        VkMemoryRequirements reqs;
        vkGetBufferMemoryRequirements(device, result.buffer, &reqs);
        VkPhysicalDeviceMemoryProperties props;
        vkGetPhysicalDeviceMemoryProperties(physical_device, &props);

        uint32_t memTypeIndex = 0;
        for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
            if ((reqs.memoryTypeBits & (1 << i)) &&
                (props.memoryTypes[i].propertyFlags & (
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
                memTypeIndex = i;
                break;
            }
        }

        VkMemoryAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = reqs.size,
            .memoryTypeIndex = memTypeIndex,
        };
        vkAllocateMemory(device, &allocInfo, nullptr, &result.memory);
        vkBindBufferMemory(device, result.buffer, result.memory, 0);

        void *mapped;
        vkMapMemory(device, result.memory, 0, reqs.size, 0, &mapped);
        memcpy(mapped, data, size);
        vkUnmapMemory(device, result.memory);

        return result;
    }

    void destroyBuffer(const VulkanBuffer &buffer) {
        vkFreeMemory(veekay::app.vk_device, buffer.memory, nullptr);
        vkDestroyBuffer(veekay::app.vk_device, buffer.buffer, nullptr);
    }

    void generateCube(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices) {
        vertices = {
            {{-0.5f, -0.5f, 0.5f}}, {{0.5f, -0.5f, 0.5f}}, {{0.5f, 0.5f, 0.5f}}, {{-0.5f, 0.5f, 0.5f}},
            {{-0.5f, -0.5f, -0.5f}}, {{0.5f, -0.5f, -0.5f}}, {{0.5f, 0.5f, -0.5f}}, {{-0.5f, 0.5f, -0.5f}}
        };
        indices = {
            0, 1, 2, 2, 3, 0, // Front
            1, 5, 6, 6, 2, 1, // Right
            5, 4, 7, 7, 6, 5, // Back
            4, 0, 3, 3, 7, 4, // Left
            3, 2, 6, 6, 7, 3, // Top
            4, 5, 1, 1, 0, 4 // Bottom
        };
    }

    void generatePyramid(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices) {
        vertices = {
            {{0.0f, 0.5f, 0.0f}}, // 0: Apex
            {{-0.5f, -0.5f, 0.5f}}, // 1: Front-Left
            {{0.5f, -0.5f, 0.5f}}, // 2: Front-Right
            {{0.5f, -0.5f, -0.5f}}, // 3: Back-Right
            {{-0.5f, -0.5f, -0.5f}} // 4: Back-Left
        };

        indices = {
            1, 2, 0, // Front face
            2, 3, 0, // Right face
            3, 4, 0, // Back face
            4, 1, 0, // Left face

            1, 4, 3,
            3, 2, 1
        };
    }

    // 3. СФЕРА (UV-Sphere)
    void generateSphere(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices, int stacks = 16,
                        int slices = 16) {
        float radius = 0.5f;

        for (int i = 0; i <= stacks; ++i) {
            float v = (float) i / (float) stacks;
            float phi = v * (float) M_PI;

            for (int j = 0; j <= slices; ++j) {
                float u = (float) j / (float) slices;
                float theta = u * 2.0f * (float) M_PI;

                float x = radius * sinf(phi) * cosf(theta);
                float y = radius * cosf(phi);
                float z = radius * sinf(phi) * sinf(theta);

                vertices.push_back({{x, -y, z}});
            }
        }

        for (int i = 0; i < stacks; ++i) {
            for (int j = 0; j < slices; ++j) {
                uint32_t first = (i * (slices + 1)) + j;
                uint32_t second = first + slices + 1;

                indices.push_back(first);
                indices.push_back(second);
                indices.push_back(first + 1);

                indices.push_back(second);
                indices.push_back(second + 1);
                indices.push_back(first + 1);
            }
        }
    }

    void createMesh(Mesh &mesh, const std::vector<Vertex> &verts, const std::vector<uint32_t> &inds) {
        mesh.index_count = (uint32_t) inds.size();
        mesh.vertex_buffer = createBuffer(verts.size() * sizeof(Vertex), (void *) verts.data(),
                                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        mesh.index_buffer = createBuffer(inds.size() * sizeof(uint32_t), (void *) inds.data(),
                                         VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    }

    void initialize() {
        VkDevice &device = veekay::app.vk_device;
        vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
        fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");

        if (!vertex_shader_module || !fragment_shader_module) {
            std::cerr << "Failed to load shaders!\n";
            veekay::app.running = false;
            return;
        }

        VkPipelineShaderStageCreateInfo stages[] = {
            {
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,
                vertex_shader_module, "main", nullptr
            },
            {
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT,
                fragment_shader_module, "main", nullptr
            }
        };

        VkVertexInputBindingDescription binding{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription attribute{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)};

        VkPipelineVertexInputStateCreateInfo vertex_input{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nullptr, 0, 1, &binding, 1, &attribute
        };
        VkPipelineInputAssemblyStateCreateInfo input_assembly{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE
        };

        VkPipelineRasterizationStateCreateInfo rasterizer{
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_FALSE,
            VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE, 0, 0, 0, 1.0f
        };

        VkPipelineMultisampleStateCreateInfo multisampling{
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0, VK_SAMPLE_COUNT_1_BIT, VK_FALSE, 0,
            nullptr, VK_FALSE, VK_FALSE
        };
        VkViewport viewport{
            0.0f, 0.0f, (float) veekay::app.window_width, (float) veekay::app.window_height, 0.0f, 1.0f
        };
        VkRect2D scissor{{0, 0}, {veekay::app.window_width, veekay::app.window_height}};
        VkPipelineViewportStateCreateInfo viewport_state{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, &viewport, 1, &scissor
        };
        VkPipelineDepthStencilStateCreateInfo depth_stencil{
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, nullptr, 0, VK_TRUE, VK_TRUE,
            VK_COMPARE_OP_LESS_OR_EQUAL, VK_FALSE, VK_FALSE, {}, {}
        };

        VkPipelineColorBlendAttachmentState color_blend_att{};
        color_blend_att.colorWriteMask = 0xF;
        VkPipelineColorBlendStateCreateInfo color_blending{
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1,
            &color_blend_att, {0, 0, 0, 0}
        };

        VkPushConstantRange push_constant{
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ShaderConstants)
        };
        VkPipelineLayoutCreateInfo layout_info{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 0, nullptr, 1, &push_constant
        };

        vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout);

        VkGraphicsPipelineCreateInfo pipeline_info{
            VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, nullptr, 0, 2, stages, &vertex_input, &input_assembly,
            nullptr, &viewport_state, &rasterizer, &multisampling, &depth_stencil, &color_blending, nullptr,
            pipeline_layout, veekay::app.vk_render_pass, 0, VK_NULL_HANDLE, 0
        };
        vkCreateGraphicsPipelines(device, nullptr, 1, &pipeline_info, nullptr, &pipeline);

        std::vector<Vertex> verts;
        std::vector<uint32_t> inds;

        generateCube(verts, inds);
        createMesh(cube_mesh, verts, inds);

        generatePyramid(verts, inds);
        createMesh(pyramid_mesh, verts, inds);

        verts.clear();
        inds.clear();
        generateSphere(verts, inds, 20, 20);
        createMesh(sphere_mesh, verts, inds);

        scene_objects.push_back({
            "Cube",
            {-2.0f, 1.0f, -6.0f},
            {1.0f, 0.2f, 0.2f}, // Красный
            &cube_mesh
        });

        scene_objects.push_back({
            "Pyramid",
            {0.0f, -1.0f, -6.0f},
            {0.2f, 1.0f, 0.2f}, // Зеленый
            &pyramid_mesh
        });

        scene_objects.push_back({
            "Sphere",
            {2.0f, 0.0f, -6.0f},
            {0.2f, 0.2f, 1.0f}, // Синий
            &sphere_mesh
        });
    }

    void shutdown() {
        VkDevice &device = veekay::app.vk_device;

        destroyBuffer(cube_mesh.vertex_buffer);
        destroyBuffer(cube_mesh.index_buffer);
        destroyBuffer(pyramid_mesh.vertex_buffer);
        destroyBuffer(pyramid_mesh.index_buffer);
        destroyBuffer(sphere_mesh.vertex_buffer);
        destroyBuffer(sphere_mesh.index_buffer);

        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyShaderModule(device, vertex_shader_module, nullptr);
        vkDestroyShaderModule(device, fragment_shader_module, nullptr);
    }

    void update(double dt) {
        if (is_animating) {
            current_time += (float) dt;
        }

        ImGui::Begin("Scene Controls");
        ImGui::Checkbox("Animate Rotation", &is_animating);
        ImGui::SliderFloat("Rotation Speed", &global_rotation_speed, 0.0f, 2.0f);
        ImGui::Separator();

        for (int i = 0; i < scene_objects.size(); ++i) {
            SceneObject &obj = scene_objects[i];
            ImGui::PushID(i);

            ImGui::Text("%s Settings:", obj.name.c_str());
            ImGui::DragFloat3("Position", &obj.position.x, 0.05f);
            ImGui::ColorEdit3("Color", &obj.color.x);

            ImGui::Separator();
            ImGui::PopID();
        }
        ImGui::End();
    }

    void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo begin_info{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr
        };
        vkBeginCommandBuffer(cmd, &begin_info);

        VkClearValue clearValues[2];
        clearValues[0].color = {{0.1f, 0.1f, 0.1f, 1.0f}};
        clearValues[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo renderPassInfo{
            VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, nullptr, veekay::app.vk_render_pass, framebuffer,
            {{0, 0}, {veekay::app.window_width, veekay::app.window_height}}, 2, clearValues
        };

        vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        float aspect = (float) veekay::app.window_width / (float) veekay::app.window_height;
        Matrix proj = perspective(camera_fov, aspect, camera_near_plane, camera_far_plane);

        for (const auto &obj: scene_objects) {
            if (!obj.mesh_ptr) continue;

            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(cmd, 0, 1, &obj.mesh_ptr->vertex_buffer.buffer, offsets);
            vkCmdBindIndexBuffer(cmd, obj.mesh_ptr->index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

            Matrix trans = translation(obj.position);
            Matrix rot = rotation_y(current_time * global_rotation_speed);

            Matrix model = multiply(rot, trans);

            Matrix mvp = multiply(model, proj);

            ShaderConstants push{mvp, obj.color};
            vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof(ShaderConstants), &push);

            vkCmdDrawIndexed(cmd, obj.mesh_ptr->index_count, 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
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
