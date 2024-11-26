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

void Object::update(vector<Object*>& all_objects)
{
    KineticState current_state{center, velocity, force / mass};
    KineticState next_state = step(prev_state, current_state);
    // 将物体的位置移动到下一步状态处，但暂时不要修改物体的速度。
    this->center = next_state.position;

    // Check collision with other objects.
    for (auto object : all_objects) {
        if(object == this) continue;
        
        for (size_t i = 0; i < mesh.edges.count(); ++i) {
            array<size_t, 2> v_indices = mesh.edge(i);
            Vector3f this_v0 = (this->model() * mesh.vertex(v_indices[0]).homogeneous()).hnormalized();
            Vector3f this_v1 = (this->model() * mesh.vertex(v_indices[1]).homogeneous()).hnormalized();

            Ray this_edge_ray = Ray{this_v0, (this_v1 - this_v0).normalized()};
            std::optional<Intersection> intersection;
            if (BVH_for_collision) intersection = object->bvh->intersect(this_edge_ray, object->mesh, object->model());
            else intersection = naive_intersect(this_edge_ray, object->mesh, object->model());
            
            if(intersection != std::nullopt && intersection->t <= (this_v1 - this_v0).norm()){
                next_state.position = current_state.position;
                // 在碰撞过程中，为什么冲量 j_r ​是沿着法向 n 而不是沿着物体的运动方向
                // 存在问题：当物体的速度，距离，以及刷新帧率满足一定条件时，交面的法向量会垂直于速度方向
                // 从而导致冲量的方向与速度方向相反，这样会导致物体的速度变为0，从而无法继续运动
                // 初步的分析是，这时候由于物体恰好接壤，因此此时作任意边到另一物体面的交点实际上都是不存在的
                // 已解决：在鉴定存在碰撞后，next_state = step(current_state, next_state);
                // 解决原因：TODO

                float impulse = -2.0f * (next_state.velocity - object->velocity).dot(intersection->normal) / (1 / mass + 1 / object->mass);
                next_state.velocity = next_state.velocity + (impulse / this->mass) * intersection->normal;
                object->velocity = object->velocity - (impulse / object->mass) * intersection->normal;

                //next_state = step(current_state, next_state); // Amazing... Who can interpret it!!!
                break;
            }
        }
    }
    // 将上一步状态赋值为当前状态，并将物体更新到下一步状态。
    this->center = next_state.position;
    this->velocity = next_state.velocity;
    this->force = next_state.acceleration * mass;
    this->prev_state = current_state;
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
