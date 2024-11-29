#include <array>
#include <limits>
#include <tuple>
#include <vector>
#include <algorithm>
#include <cmath>
#include <mutex>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <spdlog/spdlog.h>

#include "rasterizer.h"
#include "triangle.h"
#include "../utils/math.hpp"

using Eigen::Matrix4f;
using Eigen::Vector2i;
using Eigen::Vector3f;
using Eigen::Vector4f;
using std::fill;
using std::tuple;

void Rasterizer::worker_thread()
{
        while (true) {
        VertexShaderPayload payload;
        Triangle triangle;
        {
            // printf("vertex_finish = %d\n vertex_shader_output_queue.size = %ld\n",
            // Context::vertex_finish, Context::vertex_shader_output_queue.size());
            if (Context::vertex_finish && Context::vertex_shader_output_queue.empty()) {
                Context::rasterizer_finish = true;
                return;
            }
            if (Context::vertex_shader_output_queue.size() < 3) {
                printf("/n"); // Avoid the bug of the loop optimization
                continue;
            }
            std::unique_lock<std::mutex> lock(Context::vertex_queue_mutex);
            if (Context::vertex_shader_output_queue.size() < 3) {
                continue;
            }
            for (size_t vertex_count = 0; vertex_count < 3; vertex_count++) {
                payload = Context::vertex_shader_output_queue.front();
                Context::vertex_shader_output_queue.pop();
                if (vertex_count == 0) {
                    triangle.world_pos[0]    = payload.world_position;
                    triangle.viewport_pos[0] = payload.viewport_position;
                    triangle.normal[0]       = payload.normal;
                } else if (vertex_count == 1) {
                    triangle.world_pos[1]    = payload.world_position;
                    triangle.viewport_pos[1] = payload.viewport_position;
                    triangle.normal[1]       = payload.normal;
                } else {
                    triangle.world_pos[2]    = payload.world_position;
                    triangle.viewport_pos[2] = payload.viewport_position;
                    triangle.normal[2]       = payload.normal;
                }
            }
        }
        rasterize_triangle(triangle);
    }
}

float sign(Eigen::Vector2f p1, Eigen::Vector2f p2, Eigen::Vector2f p3)
{
    return (p1.x() - p3.x()) * (p2.y() - p3.y()) - (p2.x() - p3.x()) * (p1.y() - p3.y());
}

// 给定坐标(x,y)以及三角形的三个顶点坐标，判断(x,y)是否在三角形的内部
bool Rasterizer::inside_triangle(int x, int y, const Vector4f* vertices)
{
        // 将顶点转换为二维齐次坐标
    Vector3f v[3];
    for (int i = 0; i < 3; i++) 
    {
        v[i] = {vertices[i].x(), vertices[i].y(), 1.0f};
    }
    // 将点(x, y)转化为三维向量（z = 1.0）
    Vector3f p(float(x), float(y), 1.0f);

    // 计算点与三角形各边的叉积
    Vector3f edge1 = v[1] - v[0];
    Vector3f edge2 = v[2] - v[1];
    Vector3f edge3 = v[0] - v[2];

    Vector3f c_1 = p - v[0];
    Vector3f c_2 = p - v[1];
    Vector3f c_3 = p - v[2];

    // 叉积判断
    float cross1 = edge1.cross(c_1).z();
    float cross2 = edge2.cross(c_2).z();
    float cross3 = edge3.cross(c_3).z();

    // 如果所有叉积的z分量符号相同，说明点在三角形内部
    return ((cross1 >= 0 && cross2 >= 0 && cross3 >= 0) || 
           (cross1 <= 0 && cross2 <= 0 && cross3 <= 0));
}

// 给定坐标(x,y)以及三角形的三个顶点坐标，计算(x,y)对应的重心坐标[alpha, beta, gamma]
tuple<float, float, float> Rasterizer::compute_barycentric_2d(float x, float y, const Vector4f* v)
{
    float c1 = 0.f, c2 = 0.f, c3 = 0.f;
     // 三角形顶点坐标
    float x0 = v[0].x(), y0 = v[0].y();
    float x1 = v[1].x(), y1 = v[1].y();
    float x2 = v[2].x(), y2 = v[2].y();

    // 计算三角形的面积（基于叉积的大小）
    float denominator = (x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1));

    // 防止除以零的情况（退化三角形）
    if (denominator == 0.0f)
    {
        return {0.0f, 0.0f, 0.0f};
    }

    // 计算重心坐标
    c1 = (x * (y1 - y2) + (x2 - x1) * y + x1 * y2 - x2 * y1) / denominator;
    c2 = (x * (y2 - y0) + (x0 - x2) * y + x2 * y0 - x0 * y2) / denominator;
    c3 = (x * (y0 - y1) + (x1 - x0) * y + x0 * y1 - x1 * y0) / denominator;

    return {c1, c2, c3};
}

// 对顶点的某一属性插值
Vector3f Rasterizer::interpolate(float alpha, float beta, float gamma, const Eigen::Vector3f& vert1,
                                 const Eigen::Vector3f& vert2, const Eigen::Vector3f& vert3,
                                 const Eigen::Vector3f& weight, const float& Z)
{
    Vector3f interpolated_res;
    for (int i = 0; i < 3; i++) {
        interpolated_res[i] = alpha * vert1[i] / weight[0] + beta * vert2[i] / weight[1] +
                              gamma * vert3[i] / weight[2];
    }
    interpolated_res *= Z;
    return interpolated_res;
}

void Rasterizer::rasterize_triangle(Triangle& t)
{
    int min_x = (int)ceil(std::max(0.0f, std::min(std::min(t.viewport_pos[0].x(), t.viewport_pos[1].x()), t.viewport_pos[2].x())));
    int max_x = (int)floor(std::min(Context::frame_buffer.width - 1.0f, std::max(std::max(t.viewport_pos[0].x(), t.viewport_pos[1].x()), t.viewport_pos[2].x())));
    int min_y = (int)ceil(std::max(0.0f, std::min(std::min(t.viewport_pos[0].y(), t.viewport_pos[1].y()), t.viewport_pos[2].y())));
    int max_y = (int)floor(std::min(Context::frame_buffer.height - 1.0f, std::max(std::max(t.viewport_pos[0].y(), t.viewport_pos[1].y()), t.viewport_pos[2].y())));

    Vector3f weight = {t.viewport_pos[0].w(), t.viewport_pos[1].w(), t.viewport_pos[2].w()};

    for (int x = min_x; x <= max_x; x++) {
        for (int y = min_y; y <= max_y; y++) {
            // 检查像素是否在三角形内部
            if (inside_triangle(x, y, t.viewport_pos)) {
                auto [alpha, beta, gamma] = compute_barycentric_2d(x, y, t.viewport_pos);

                // 进行深度校正插值
                float Z = 1 / (alpha / t.viewport_pos[0].w() + beta / t.viewport_pos[1].w() + gamma / t.viewport_pos[2].w());
                Z = (alpha * t.viewport_pos[0].z() / t.viewport_pos[0].w() +
                     beta * t.viewport_pos[1].z() / t.viewport_pos[1].w() +
                     gamma * t.viewport_pos[2].z() / t.viewport_pos[2].w()) * Z;

                // 计算深度缓冲区索引
                int index = (Context::frame_buffer.height - 1 - y) * Context::frame_buffer.width + x;

                // 更新深度缓冲区和插值顶点属性
                if (Z <= Context::frame_buffer.depth_buffer[index]) {
                    Context::frame_buffer.depth_buffer[index] = Z;

                    // 插值顶点位置和法线
                    Vector3f interpolated_pos = interpolate(alpha, beta, gamma,
                                                             t.world_pos[0].head(3), t.world_pos[1].head(3), t.world_pos[2].head(3),
                                                             weight, Z);
                    Vector3f interpolated_normal = interpolate(alpha, beta, gamma,
                                                                t.normal[0], t.normal[1], t.normal[2],
                                                                weight, Z);

                    // 准备片段属性
                    FragmentShaderPayload payload = {
                        interpolated_pos, interpolated_normal, x, y, Z, Vector3f::Zero()};

                    // 加入片段队列
                    std::unique_lock<std::mutex> lock(Context::rasterizer_queue_mutex);
                    Context::rasterizer_output_queue.push(payload);
                }
            }
        }
    }
}
