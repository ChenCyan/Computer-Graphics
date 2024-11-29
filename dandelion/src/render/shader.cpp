#include "rasterizer_renderer.h"
#include "../utils/math.hpp"
#include <cstdio>

#ifdef _WIN32
#undef min
#undef max
#endif

using Eigen::Vector3f;
using Eigen::Vector4f;

// vertex shader
VertexShaderPayload vertex_shader(const VertexShaderPayload& payload)
{
    VertexShaderPayload output_payload = payload;
    output_payload.world_position   =   payload.world_position;
    // 投影到裁剪空间
    Eigen::Vector4f clip_position = Uniforms::MVP * payload.world_position;
    // TO DO : how to convert model to world?

    // 透视除法到 NDC
    Eigen::Vector4f ndc_position(clip_position.x() / clip_position.w(), 
                                 clip_position.y() / clip_position.w(), 
                                 clip_position.z() / clip_position.w(), 
                                 1.0f);

    // 视口变换
    Eigen::Matrix4f viewport_matrix;
    viewport_matrix << 
        Uniforms::width / 2.0f, 0, 0, Uniforms::width / 2.0f,
        0, Uniforms::height / 2.0f, 0, Uniforms::height / 2.0f,
        0, 0, 0.5f, 0.5f,
        0, 0, 0, 1;
    //from (-1,1) to (0,1)
    output_payload.viewport_position = viewport_matrix * ndc_position;

    // 法线变换（逆转置矩阵）
    Vector4f normal_4_dimension = {payload.normal[0], payload.normal[1], payload.normal[2], 0};
    output_payload.normal            = (Uniforms::inv_trans_M * normal_4_dimension).head(3).normalized();

    return output_payload;
}

Vector3f phong_fragment_shader(const FragmentShaderPayload& payload, const GL::Material& material,
                               const std::list<Light>& lights, const Camera& camera)
{
     // these lines below are just for compiling and can be deleted
    (void)payload;
    (void)material;
    (void)lights;
    (void)camera;
    // these lines above are just for compiling and can be deleted

    Vector3f result = {0, 0, 0};

    // ka,kd,ks can be got from material.ambient,material.diffuse,material.specular
    Eigen::Vector3f ka  =   material.ambient;
    Eigen::Vector3f kd  =   material.diffuse;
    Eigen::Vector3f ks  =   material.specular;
    // set ambient light intensity
    float ambient_light_intensity = 0.2f;
    result = result +  ka * ambient_light_intensity;

    for(const auto& light : lights)
    {
        // Light Direction
        Eigen::Vector3f light_direction = (light.position - payload.world_normal).normalized();
        // View Direction
        Eigen::Vector3f view_direction  = (camera.position - payload.world_normal).normalized();
        // Half Vector
        Eigen::Vector3f half_vector     = (light_direction + view_direction).normalized();
        // Light Attenuation

        // Diffuse
        Eigen::Vector3f normalized_normal = payload.world_normal.normalized();
        float dist_to_light             = (light.position - payload.world_pos).norm();
        float attenuation               = 1.0f / (dist_to_light * dist_to_light);
        float diffuse_factor            = std::max(0.0f, normalized_normal.dot(light_direction));
        Eigen::Vector3f diffuse_light = light.intensity * attenuation * kd * diffuse_factor;

        // Specular
        float specular_factor = std::pow(std::max(0.0f,normalized_normal.dot(half_vector)), material.shininess);
        Vector3f specular_light = light.intensity * attenuation * ks * specular_factor;

        result = result + diffuse_light;
        result = result + specular_light;
    }
    // 限制渲染结果并调整至 255 的范围
    result.x() = std::min(result.x(), 1.0f) * 255.0f;
    result.y() = std::min(result.y(), 1.0f) * 255.0f;
    result.z() = std::min(result.z(), 1.0f) * 255.0f;

    return result;
}