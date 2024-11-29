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
    float vertical_fov = radians(camera.fov_y_degrees);  // 垂直视场角（单位：弧度）
    float height_on_plane = 2.0f * tan(vertical_fov / 2.0f);  // 图像平面高度
    float width_to_height_ratio = static_cast<float>(width) / height;  // 图像宽高比
    float width_on_plane = height_on_plane * width_to_height_ratio;  // 图像平面宽度

    // 将像素坐标转化为标准化设备坐标（NDC）
    float ndc_x = (static_cast<float>(x) + 0.5f) / width;
    float ndc_y = (static_cast<float>(y) + 0.5f) / height;

    // 将NDC坐标映射到屏幕空间坐标
    float screen_x = 2.0f * ndc_x - 1.0f;
    float screen_y = 1.0f - 2.0f * ndc_y;

    // 计算相机空间的像素坐标
    float camera_x = screen_x * width_on_plane / 2.0f;
    float camera_y = screen_y * height_on_plane / 2.0f;

    // 创建像素在相机空间中的坐标（x, y, -depth）
    Vector4f pixel_in_camera_space(camera_x, camera_y, -depth, 1.0f);

    // 计算视图矩阵的逆矩阵
    Matrix4f view_matrix_inv = camera.view().inverse();

    // 将像素坐标从相机空间转换到世界空间
    Vector4f pixel_in_world_space = view_matrix_inv * pixel_in_camera_space;

    // 计算光线的起点和方向
    Vector3f ray_origin = camera.position;
    Vector3f ray_direction = (pixel_in_world_space.head<3>() / pixel_in_world_space.w() - ray_origin).normalized();

    // 返回生成的光线
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

std::optional<Intersection> intersection_result;  // 用于存储相交的最终结果
float closest_t = std::numeric_limits<float>::infinity();  // 初始化为正无穷，表示尚未找到交点

// 遍历每个三角形面
for (size_t idx = 0; idx < mesh.faces.count(); ++idx) {
    const auto& triangle = mesh.face(idx);  // 获取当前的三角形

    // 将三角形的三个顶点从模型空间转换到世界空间
    Vector3f vertex1 = (model * mesh.vertex(triangle[0]).homogeneous()).hnormalized();
    Vector3f vertex2 = (model * mesh.vertex(triangle[1]).homogeneous()).hnormalized();
    Vector3f vertex3 = (model * mesh.vertex(triangle[2]).homogeneous()).hnormalized();

    // 计算三角形的法向量
    Vector3f triangle_normal = (vertex2 - vertex1).cross(vertex3 - vertex1).normalized();
    float dot_product = triangle_normal.dot(ray.direction);

    // 如果光线平行于平面，跳过该面
    if (std::abs(dot_product) < eps) continue;

    // 计算光线与平面的交点 t 值
    float t_value = (vertex1 - ray.origin).dot(triangle_normal) / dot_product;
    if (t_value < eps || t_value >= closest_t) continue;

    // 计算交点位置
    Vector3f intersection_point = ray.origin + t_value * ray.direction;

    // 判断交点是否在三角形内部（通过逐边法判断）
    bool is_inside = 
        triangle_normal.dot((vertex2 - vertex1).cross(intersection_point - vertex1)) >= 0 &&
        triangle_normal.dot((vertex3 - vertex2).cross(intersection_point - vertex2)) >= 0 &&
        triangle_normal.dot((vertex1 - vertex3).cross(intersection_point - vertex3)) >= 0;

    // 如果交点在三角形内并且 t 值为最近的交点，则更新结果
    if (is_inside && t_value < closest_t) {
        Intersection temp_result;
        closest_t = t_value;
        temp_result.t = t_value;
        temp_result.barycentric_coord = intersection_point;
        temp_result.normal = triangle_normal;
        intersection_result = temp_result;
    }
}

// 如果找到相交点，返回交点信息，否则返回空
return (closest_t < std::numeric_limits<float>::infinity()) ? intersection_result : std::nullopt;

}
