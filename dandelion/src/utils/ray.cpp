#include "ray.h"

#include <cmath>
#include <array>

#include <Eigen/Dense>
#include <spdlog/spdlog.h>

#include "../utils/math.hpp"

using Eigen::Matrix3f;
using Eigen::Matrix4f;
using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::Vector4f;
using std::numeric_limits;
using std::optional;
using std::size_t;

constexpr float infinity = 1e5f;
constexpr float eps      = 1e-5f;

Intersection::Intersection() : t(numeric_limits<float>::infinity()), face_index(0)
{

}

Ray generate_ray(int width, int height, int x, int y, Camera& camera, float depth)
{
     float fov_y = radians(camera.fov_y_degrees);
    float image_plane_height = 2.0f * tan(fov_y / 2.0f);
    float aspect_ratio = static_cast<float>(width) / height;
    float image_plane_width = image_plane_height * aspect_ratio;

    float pixel_ndc_x = (static_cast<float>(x) + 0.5f) / width;
    float pixel_ndc_y = (static_cast<float>(y) + 0.5f) / height;

    float pixel_screen_x = 2.0f * pixel_ndc_x - 1.0f;
    float pixel_screen_y = 1.0f - 2.0f * pixel_ndc_y;

    float pixel_camera_x = pixel_screen_x * image_plane_width / 2.0f;
    float pixel_camera_y = pixel_screen_y * image_plane_height / 2.0f;

    Vector4f pixel_camera_space(pixel_camera_x, pixel_camera_y, -depth, 1.0f);
    Matrix4f inv_view = camera.view().inverse();
    Vector4f pixel_world_space = inv_view * pixel_camera_space;

    Vector3f ray_origin = camera.position;
    Vector3f ray_direction = (pixel_world_space.head<3>() / pixel_world_space.w() - ray_origin).normalized();
    return {ray_origin, ray_direction};
}

optional<Intersection> ray_triangle_intersect(const Ray& ray, const GL::Mesh& mesh, size_t index) {
    // 获取三角形面
    const auto& face = mesh.face(index);

    // 获取三角形的三个顶点
    Eigen::Vector3f a = mesh.vertex(face[0]);
    Eigen::Vector3f b = mesh.vertex(face[1]);
    Eigen::Vector3f c = mesh.vertex(face[2]);

    // 计算三角形面法向量
    Eigen::Vector3f normal = (b - a).cross(c - a).normalized();
    float normal_dot_dir = normal.dot(ray.direction);

    // 检查射线是否与平面平行
    if (std::abs(normal_dot_dir) < eps) {
        return std::nullopt;
    }

    // 计算射线和平面的交点参数 t
    float t = (a - ray.origin).dot(normal) / normal_dot_dir;
    if (t < eps) {
        return std::nullopt; // 如果 t < eps，交点在射线的起点之前
    }

    // 计算射线与平面的交点 P
    Eigen::Vector3f P = ray.origin + t * ray.direction;

    // 使用叉积法逐边测试交点是否在三角形内部
    if (normal.dot((b - a).cross(P - a)) < 0 ||
        normal.dot((c - b).cross(P - b)) < 0 ||
        normal.dot((a - c).cross(P - c)) < 0) {
        return std::nullopt; // 如果交点不在三角形内部，返回空值
    }

    // 计算重心坐标（质心插值）
    float areaABC = normal.norm(); // 三角形的总面积为法向量的模
    float areaPBC = (b - P).cross(c - P).norm(); // PBC 子三角形面积
    float areaPAC = (a - P).cross(c - P).norm(); // PAC 子三角形面积
    float areaPAB = (a - P).cross(b - P).norm(); // PAB 子三角形面积

    // 归一化重心坐标
    float u = areaPBC / areaABC;
    float v = areaPAC / areaABC;
    float w = areaPAB / areaABC;

    // 返回交点信息
    Intersection result;
    result.t = t;                               // 射线参数 t
    result.face_index = index;                  // 三角形索引
    result.barycentric_coord = {u, v, w};       // 重心坐标
    result.normal = normal;                     // 三角形的法向量
    return result;
}


optional<Intersection> naive_intersect(const Ray& ray, const GL::Mesh& mesh, const Matrix4f model)
{
    // these lines below are just for compiling and can be deleted
    (void)ray;
    (void)model;
    // these lines above are just for compiling and can be deleted
    optional<Intersection> result;  // 存储最终的相交结果
    float min_t_in_scoop = std::numeric_limits<float>::infinity(); // 初始化为正无穷，以找到最近的相交点

    // 遍历网格的每个三角形面
    for (size_t i = 0; i < mesh.faces.count(); i++) {
        const auto& face = mesh.face(i);  // 获取第 i 个三角形面

        // 获取三角形的三个顶点并从模型空间转换到世界空间
        Vector3f vertex_a = (model * mesh.vertex(face[0]).homogeneous()).hnormalized();
        Vector3f vertex_b = (model * mesh.vertex(face[1]).homogeneous()).hnormalized();
        Vector3f vertex_c = (model * mesh.vertex(face[2]).homogeneous()).hnormalized();

        // 计算三角形面法向量
        Vector3f normal = (vertex_b - vertex_a).cross(vertex_c - vertex_a).normalized();
        float normal_dot_dir = normal.dot(ray.direction);
        //printf("normal_dot_dir = %lf\n",normal_dot_dir);
        // 检查光线是否与平面平行
        //printf("vertex_a = (%lf,%lf,%lf)\n  ray.origin = (%lf,%lf,%lf)\n",vertex_a[0],vertex_a[1],vertex_a[2],ray.origin[0],ray.origin[1],ray.origin[2]);
        if (std::abs(normal_dot_dir) < eps) continue;

        // 计算光线与平面相交的 t 值
        float t = (vertex_a - ray.origin).dot(normal) / normal_dot_dir;
        if (t < eps || t >= min_t_in_scoop) continue;

        // 计算光线与平面的交点
        Vector3f cross_point = ray.origin + t * ray.direction;

        // 使用叉积法判断交点是否在三角形内部（逐边测试）
        bool inside_triangle = 
            normal.dot((vertex_b - vertex_a).cross(cross_point - vertex_a)) >= 0 &&
            normal.dot((vertex_c - vertex_b).cross(cross_point - vertex_b)) >= 0 &&
            normal.dot((vertex_a - vertex_c).cross(cross_point - vertex_c)) >= 0;
        
        // 如果交点在三角形内部且 t 为最近的相交距离，则更新结果
        if (inside_triangle && t < min_t_in_scoop) {
            Intersection _result;
            min_t_in_scoop = t;
            _result.t = t;
            _result.barycentric_coord = cross_point;
            _result.normal = normal;
        result = _result;
        }
    }

    // 如果找到相交点，则返回相交结果，否则返回空
    return (min_t_in_scoop < std::numeric_limits<float>::infinity()) ? result : std::nullopt;
}
