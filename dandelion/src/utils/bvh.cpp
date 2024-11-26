#include "bvh.h"

#include <cassert>
#include <iostream>
#include <optional>

#include <Eigen/Geometry>
#include "formatter.hpp"
#include <spdlog/spdlog.h>

#include "math.hpp"

using Eigen::Vector3f;
using std::optional;
using std::vector;
int number = 0;
void print_bvh_tree(BVHNode* node, int depth = 0)
{
    if (!node)
        return;
    // 打印当前节点信息
    std::string indent(depth * 2, ' '); // 缩进，表示层级关系
    std::cout << indent << "Node at depth " << depth << "\n";
    std::cout << indent << "  Face Index: " << node->face_idx << "\n";
    std::cout << indent << "  AABB: [(" << node->aabb.p_min.x() << ", " << node->aabb.p_min.y()
              << ", " << node->aabb.p_min.z() << "), (" << node->aabb.p_max.x() << ", "
              << node->aabb.p_max.y() << ", " << node->aabb.p_max.z() << ")]\n";
    if (node->face_idx != 0)
        number++;
    // 如果有子节点，继续递归打印
    if (node->left || node->right) {
        std::cout << indent << "  Left Child:\n";
        print_bvh_tree(node->left, depth + 1);
        std::cout << indent << "  Right Child:\n";
        print_bvh_tree(node->right, depth + 1);
    }
}
BVHNode::BVHNode() : left(nullptr), right(nullptr), face_idx(0)
{
}

BVH::BVH(const GL::Mesh& mesh) : root(nullptr), mesh(mesh)
{
}

// 建立bvh，将需要建立BVH的图元索引初始化
void BVH::build()
{
    if (mesh.faces.count() == 0) {
        root = nullptr;
        return;
    }
    //printf("call build\n");
    primitives.resize(mesh.faces.count());
    for (size_t i = 0; i < mesh.faces.count(); i++) primitives[i] = i;

    root = recursively_build(primitives);
    // print_bvh_tree(root);
    // printf("the number of child node is %d\n",number);
    return;
}
// 删除bvh
int num = 0;
void BVH::recursively_delete(BVHNode* node)
{
    if (node == nullptr)
        return;
    recursively_delete(node->left);
    recursively_delete(node->right);
    node = nullptr;
}
// 统计BVH树建立的节点个数
size_t BVH::count_nodes(BVHNode* node)
{
    if (node == nullptr)
        return 0;
    else
        return count_nodes(node->left) + count_nodes(node->right) + 1;
}
// 递归建立BVH
BVHNode* BVH::recursively_build(vector<size_t> faces_idx)
{
    num++;
    // printf("use recursively_build %d times \n",num);
    if (faces_idx.size() == 0)
        return nullptr;
    if (faces_idx.size() == 1) {
        // printf("child\n");
        BVHNode* leaf  = new BVHNode();
        leaf->face_idx = faces_idx[0];
        leaf->left     = nullptr;
        leaf->right    = nullptr;
        leaf->aabb     = get_aabb(mesh, faces_idx[0]);
        // printf("child.idx=%ld\n",faces_idx[0]);
        /*std::cout <<  "  AABB: [("
              << leaf->aabb.p_min.x() << ", " << leaf->aabb.p_min.y() << ", " <<
           leaf->aabb.p_min.z() << "), ("
              << leaf->aabb.p_max.x() << ", " << leaf->aabb.p_max.y() << ", " <<
           leaf->aabb.p_max.z() << ")]\n";
        */
        return leaf;
    }

    if (faces_idx.size() == 2) {
        BVHNode* node       = new BVHNode();
        BVHNode* left_leaf  = recursively_build({faces_idx[0]});
        BVHNode* right_leaf = recursively_build({faces_idx[1]});
        node->left          = left_leaf;
        // printf("left child.idx=%ld\n",node->left->face_idx);
        node->right = right_leaf;
        // printf("right child.idx=%ld\n",node->right->face_idx);
        node->aabb     = union_AABB(left_leaf->aabb, right_leaf->aabb); // 合并左右 AABB
        node->face_idx = 0; // 中间节点不存储面片索引
        return node;
    }

    BVHNode* node = new BVHNode();

    AABB aabb;
    for (size_t i = 0; i < faces_idx.size(); i++) {
        aabb = union_AABB(aabb, get_aabb(mesh, faces_idx[i]));
    }
    // if faces_idx.size()==1: return node;
    // if faces_idx.size()==2: recursively_build() & union_AABB(node->left->aabb,
    // node->right->aabb); else:
    // choose the longest dimension among x,y,z
    // devide the primitives into two along the longest dimension
    // recursively_build() & union_AABB(node->left->aabb, node->right->aabb)
    node->aabb    = aabb;
    int split_dim = aabb.max_extent(); // 找到最长的一维 (0: x, 1: y, 2: z)
                                       // 按照选定维度对面片索引排序
    std::sort(faces_idx.begin(), faces_idx.end(), [&](size_t f1, size_t f2) {
        Vector3f centroid1 = get_aabb(mesh, f1).centroid();
        Vector3f centroid2 = get_aabb(mesh, f2).centroid();
        return centroid1[split_dim] < centroid2[split_dim];
    });

    // 将面片索引分为两组
    size_t mid = faces_idx.size() / 2;
    vector<size_t> left_faces(faces_idx.begin(), faces_idx.begin() + mid);
    vector<size_t> right_faces(faces_idx.begin() + mid, faces_idx.end());

    // 递归构建左右子节点
    node->left = recursively_build(left_faces);
    // printf("left child.idx=%ld\n",node->left->face_idx);
    node->right = recursively_build(right_faces);
    // printf("right child.idx=%ld\n",node->right->face_idx);
    //  当前节点不是叶子节点，设置 face_idx 为 0
    node->face_idx = 0;

    return node;
}
// 使用BVH求交
optional<Intersection> BVH::intersect(const Ray& ray, [[maybe_unused]] const GL::Mesh& mesh,
                                      const Eigen::Matrix4f obj_model)
{
    // print_bvh_tree(root);
    this->model = obj_model;
    optional<Intersection> isect;
    if (!root) {
        isect = std::nullopt;
        return isect;
    }
    // print_bvh_tree(root);
    isect = ray_node_intersect(root, ray);
    return isect;
}
// 发射的射线与当前节点求交，并递归获取最终的求交结果
optional<Intersection> BVH::ray_node_intersect(BVHNode* node, const Ray& ray) const
{
    optional<Intersection> isect;
    // The node intersection is performed in the model coordinate system.
    // Therefore, the ray needs to be transformed into the model coordinate system.
    // The intersection attributes returned are all in the model coordinate system.
    // Therefore, They are need to be converted to the world coordinate system.    
    // If the model shrinks, the value of t will also change.
    // The change of t can be solved by intersection point changing simultaneously

    // transform ray to model coordinate system
    Ray model_ray = ray;

    Eigen::Matrix4f world_to_model = model.inverse();
    Vector3f model_origin = (world_to_model*ray.origin.homogeneous()).hnormalized();
    Vector3f model_target = (world_to_model * (Eigen::Vector4f() << (ray.origin + ray.direction), 1.0f).finished()).head<3>();
    //Vector3f model_direction = (world_to_model*ray.direction.homogeneous()).hnormalized();
    model_ray.origin = model_origin;
    model_ray.direction = (model_target - model_origin).normalized();

    // 计算射线方向的倒数
    Vector3f inv_dir(1 / model_ray.direction.x(), 1 / model_ray.direction.y(),
                     1 / model_ray.direction.z());

    // 判断射线在各个坐标轴上的正负方向，1为正向，0为负向
    std::array<int, 3> dir_is_neg = {(model_ray.direction.x() <= 0) ? 0 : 1,
                                     (model_ray.direction.y() <= 0) ? 0 : 1,
                                     (model_ray.direction.z() <= 0) ? 0 : 1};

    // 检查射线是否与当前节点的包围盒相交
    if (!node->aabb.intersect(model_ray, inv_dir, dir_is_neg)) {
        // 如果不相交，直接返回空值
        return isect;
    }

    // 如果当前节点是叶子节点，则求解射线与三角形的交点
    if (!node->left && !node->right) {
        optional<Intersection> result = ray_triangle_intersect(model_ray, mesh, node->face_idx);
        if (!result.has_value()) {
            return std::nullopt;
        }
        // 将交点属性从模型坐标系转换到世界坐标系
        Intersection result_world = *result;
        result_world.normal = (model.inverse().transpose() * result_world.normal.homogeneous()).hnormalized();
        result_world.normal.normalize();
        return result_world;
    }

    // 如果当前节点不是叶子节点，则递归地在左右子节点上进行射线求交
    optional<Intersection> left_result = ray_node_intersect(node->left, ray);
    optional<Intersection> right_result = ray_node_intersect(node->right, ray);

    // 返回最近的交点
    if (left_result.has_value() && (!right_result.has_value() || left_result->t < right_result->t)) {
        return left_result;
    }
    return right_result;

}
/*optional<Intersection> BVH::ray_node_intersect(BVHNode* node, const Ray& ray) const
{
    optional<Intersection> isect;
    // The node intersection is performed in the model coordinate system.
    // Therefore, the ray needs to be transformed into the model coordinate system.
    // The intersection attributes returned are all in the model coordinate system.
    // Therefore, They are need to be converted to the world coordinate system.    
    // If the model shrinks, the value of t will also change.
    // The change of t can be solved by intersection point changing simultaneously

    // transform ray to model coordinate system
    Ray model_ray = ray;

    Eigen::Matrix4f world_to_model = model.inverse();
    Vector3f model_origin = (world_to_model * (Eigen::Vector4f() << ray.origin, 1.0f).finished()).head<3>();
    Vector3f model_target = (world_to_model * (Eigen::Vector4f() << (ray.origin + ray.direction), 1.0f).finished()).head<3>();
    model_ray.origin = model_origin;
    model_ray.direction = (model_target - model_origin).normalized();

    // 计算射线方向的倒数
    Vector3f inv_dir(1 / model_ray.direction.x(), 1 / model_ray.direction.y(),
                     1 / model_ray.direction.z());

    // 判断射线在各个坐标轴上的正负方向，1为正向，0为负向
    std::array<int, 3> dir_is_neg = {(model_ray.direction.x() <= 0) ? 0 : 1,
                                     (model_ray.direction.y() <= 0) ? 0 : 1,
                                     (model_ray.direction.z() <= 0) ? 0 : 1};

    // 检查射线是否与当前节点的包围盒相交
    if (!node->aabb.intersect(model_ray, inv_dir, dir_is_neg)) {
        // 如果不相交，直接返回空值
        return isect;
    }

    // 如果当前节点是叶子节点，则求解射线与三角形的交点
    if (!node->left && !node->right) {
        optional<Intersection> result = ray_triangle_intersect(model_ray, mesh, node->face_idx);
        if (!result.has_value()) {
            return std::nullopt;
        }
        // 将交点属性从模型坐标系转换到世界坐标系
        Intersection result_world = *result;
        result_world.normal = (model.inverse().transpose() * Eigen::Vector4f(result_world.normal.x(), result_world.normal.y(), result_world.normal.z(), 0.0f)).head<3>().normalized();
        return result_world;
    }

    // 如果当前节点不是叶子节点，则递归地在左右子节点上进行射线求交
    optional<Intersection> left_result = ray_node_intersect(node->left, ray);
    optional<Intersection> right_result = ray_node_intersect(node->right, ray);

    // 返回最近的交点
    if (left_result.has_value() && (!right_result.has_value() || left_result->t < right_result->t)) {
        return left_result;
    }
    return right_result;

}*/