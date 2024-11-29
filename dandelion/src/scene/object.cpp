#include "object.h"

#include <array>
#include <optional>

#ifdef _WIN32
#include <Windows.h>
#endif
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fmt/format.h>
#include <iostream>
#include "../utils/math.hpp"
#include "../utils/ray.h"
#include "../simulation/solver.h"
#include "../utils/logger.h"

using Eigen::Matrix4f;
using Eigen::Quaternionf;
using Eigen::Vector3f;
using std::array;
using std::make_unique;
using std::optional;
using std::string;
using std::vector;
int j=0;
/*void print_bvh_tree(BVHNode* node, int depth = 0) {
    if (!node) return;

    // 打印当前节点信息
    std::string indent(depth * 2, ' '); // 缩进，表示层级关系
    std::cout << indent << "Node at depth " << depth << "\n";
    std::cout << indent << "  Face Index: " << node->face_idx << "\n";
    std::cout << indent << "  AABB: [(" 
              << node->aabb.p_min.x() << ", " << node->aabb.p_min.y() << ", " << node->aabb.p_min.z() << "), ("
              << node->aabb.p_max.x() << ", " << node->aabb.p_max.y() << ", " << node->aabb.p_max.z() << ")]\n";

    // 如果有子节点，继续递归打印
    if (node->left || node->right) {
        std::cout << indent << "  Left Child:\n";
        print_bvh_tree(node->left, depth + 1);
        std::cout << indent << "  Right Child:\n";
        print_bvh_tree(node->right, depth + 1);
    }
}*/
bool Object::BVH_for_collision   = false;
size_t Object::next_available_id = 0;
std::function<KineticState(const KineticState&, const KineticState&)> Object::step =
    forward_euler_step;

Object::Object(const string& object_name)
    : name(object_name), center(0.0f, 0.0f, 0.0f), scaling(1.0f, 1.0f, 1.0f),
      rotation(1.0f, 0.0f, 0.0f, 0.0f), velocity(0.0f, 0.0f, 0.0f), force(0.0f, 0.0f, 0.0f),
      mass(1.0f), BVH_boxes("BVH", GL::Mesh::highlight_wireframe_color)
{
    visible  = true;
    modified = false;
    id       = next_available_id;
    ++next_available_id;
    bvh                      = make_unique<BVH>(mesh);
    const string logger_name = fmt::format("{} (Object ID: {})", name, id);
    logger                   = get_logger(logger_name);
}

Matrix4f Object::model()
{
Matrix4f translation_m = Matrix4f::Identity();
translation_m(0, 3) = center(0);
translation_m(1, 3) = center(1); 
translation_m(2, 3) = center(2);

Matrix4f scaling_m;
scaling_m << scaling(0), 0, 0, 0,
             0, scaling(1), 0, 0,
             0, 0, scaling(2), 0,
             0, 0, 0, 1;

const Quaternionf& r = rotation;
auto [x_angle, y_angle, z_angle] = quaternion_to_ZYX_euler(r.w(), r.x(), r.y(), r.z());

float x_angle_ra_cos = cos(radians(x_angle));
float x_angle_ra_sin = sin(radians(x_angle));
float y_angle_ra_cos = cos(radians(y_angle));
float y_angle_ra_sin = sin(radians(y_angle));
float z_angle_ra_cos = cos(radians(z_angle));
float z_angle_ra_sin = sin(radians(z_angle));

// 正确初始化 rotation_mx
Matrix4f rotation_mx;
rotation_mx << 1, 0, 0, 0,
               0, x_angle_ra_cos, -x_angle_ra_sin, 0,
               0, x_angle_ra_sin, x_angle_ra_cos, 0,
               0, 0, 0, 1;

// 正确初始化 rotation_my
Matrix4f rotation_my;
rotation_my << y_angle_ra_cos, 0, y_angle_ra_sin, 0,
               0, 1, 0, 0,
               -y_angle_ra_sin, 0, y_angle_ra_cos, 0,
               0, 0, 0, 1;

// 正确初始化 rotation_mz
Matrix4f rotation_mz;
rotation_mz << z_angle_ra_cos, -z_angle_ra_sin, 0, 0,
               z_angle_ra_sin, z_angle_ra_cos, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1;

// 构建最终的变换矩阵
Matrix4f change_matrix = translation_m * rotation_mx * rotation_my * rotation_mz * scaling_m;
return change_matrix;

}

void Object::update(std::vector<Object*>& all_objects)
{
    // 获取当前状态与下一状态
    KineticState current_status{center, velocity, force / mass};
    KineticState next_status = step(prev_state, current_status);

    // 更新物体位置
    this->center = next_status.position;

    // 遍历所有物体检查碰撞
    for (auto& obj : all_objects) {
        if (obj == this) continue;

        for (size_t i = 0; i < mesh.edges.count(); ++i) {
            auto edge_vertices = mesh.edge(i);
            Vector3f vertex_start = (this->model() * mesh.vertex(edge_vertices[0]).homogeneous()).hnormalized();
            Vector3f vertex_end = (this->model() * mesh.vertex(edge_vertices[1]).homogeneous()).hnormalized();

            // 创建射线表示当前边
            Ray edge_ray{vertex_start, (vertex_end - vertex_start).normalized()};
            std::optional<Intersection> collision_point;
            if (BVH_for_collision) {
                collision_point = obj->bvh->intersect(edge_ray, obj->mesh, obj->model());
            } else {
                collision_point = naive_intersect(edge_ray, obj->mesh, obj->model());
            }

            // 如果发生碰撞，并且距离足够小
            if (collision_point && collision_point->t <= (vertex_end - vertex_start).norm()) {
                // 碰撞发生时重置位置并计算冲量
                next_status.position = current_status.position;

                // 计算冲量，沿法向量更新速度
                float impulse_value = -2.0f * (next_status.velocity - obj->velocity).dot(collision_point->normal) / 
                                      (1 / mass + 1 / obj->mass);
                next_status.velocity += (impulse_value / this->mass) * collision_point->normal;
                obj->velocity -= (impulse_value / obj->mass) * collision_point->normal;

                break;
            }
        }
    }

    // 更新物体的最终状态
    this->center = next_status.position;
    this->velocity = next_status.velocity;
    this->force = next_status.acceleration * mass;
    this->prev_state = current_status;
}




void Object::render(const Shader& shader, WorkingMode mode, bool selected)
{
    if (modified) {
        mesh.VAO.bind();
        mesh.vertices.to_gpu();
        mesh.normals.to_gpu();
        mesh.edges.to_gpu();
        mesh.edges.release();
        mesh.faces.to_gpu();
        mesh.faces.release();
        mesh.VAO.release();
    }
    modified = false;
    // Render faces anyway.
    unsigned int element_flags = GL::Mesh::faces_flag;
    if (mode == WorkingMode::MODEL) {
        // For *Model* mode, only the selected object is rendered at the center in the world.
        // So the model transform is the identity matrix.
        shader.set_uniform("model", I4f);
        shader.set_uniform("normal_transform", I4f);
        element_flags |= GL::Mesh::vertices_flag;
        element_flags |= GL::Mesh::edges_flag;
    } else {
        Matrix4f model = this->model();
        shader.set_uniform("model", model);
        shader.set_uniform("normal_transform", (Matrix4f)(model.inverse().transpose()));
    }
    // Render edges of the selected object for modes with picking enabled.
    if (check_picking_enabled(mode) && selected) {
        element_flags |= GL::Mesh::edges_flag;
    }
    mesh.render(shader, element_flags);
}

void Object::rebuild_BVH()
{
    bvh->recursively_delete(bvh->root);
    bvh->build();
    BVH_boxes.clear();
    refresh_BVH_boxes(bvh->root);
    BVH_boxes.to_gpu();
}

void Object::refresh_BVH_boxes(BVHNode* node)
{
    if (node == nullptr) {
        return;
    }
    BVH_boxes.add_AABB(node->aabb.p_min, node->aabb.p_max);
    refresh_BVH_boxes(node->left);
    refresh_BVH_boxes(node->right);
}
