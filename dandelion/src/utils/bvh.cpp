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

    primitives.resize(mesh.faces.count());
    for (size_t i = 0; i < mesh.faces.count(); i++) primitives[i] = i;

    root = recursively_build(primitives);
    return;
}
// 删除bvh
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

    if (faces_idx.size() <= 1) {
        BVHNode* leaf = new BVHNode();
        leaf->face_idx = faces_idx.empty() ? 0 : faces_idx[0];
        leaf->left = nullptr;
        leaf->right = nullptr;
        leaf->aabb = faces_idx.empty() ? AABB() : get_aabb(mesh, faces_idx[0]);
        return leaf;
    }

    if (faces_idx.size() == 2) {
        BVHNode* node = new BVHNode();
        BVHNode* left_leaf = recursively_build({faces_idx[0]});
        BVHNode* right_leaf = recursively_build({faces_idx[1]});
        node->left = left_leaf;
        node->right = right_leaf;
        node->aabb = union_AABB(left_leaf->aabb, right_leaf->aabb); // 合并左右 AABB
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
    node->aabb = aabb;
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
    node->right = recursively_build(right_faces);

    // 当前节点不是叶子节点，设置 face_idx 为 0
    node->face_idx = 0;


    return node;
}
// 使用BVH求交
optional<Intersection> BVH::intersect(const Ray& ray, [[maybe_unused]] const GL::Mesh& mesh,
                                      const Eigen::Matrix4f obj_model)
{
    model = obj_model;
    optional<Intersection> isect;
    if (!root) {
        isect = std::nullopt;
        return isect;
    }
    isect = ray_node_intersect(root, ray);

    if (isect.has_value())
    {
        isect->barycentric_coord = (obj_model * isect->barycentric_coord.homogeneous()).hnormalized();
        isect->normal = (obj_model * isect->normal.homogeneous()).hnormalized();
    }
    return isect;
}
// 发射的射线与当前节点求交，并递归获取最终的求交结果
optional<Intersection> BVH::ray_node_intersect(BVHNode* node, const Ray& ray) const
{
    // these lines below are just for compiling and can be deleted
    (void)ray;
    (void)node;
    // these lines above are just for compiling and can be deleted

    optional<Intersection> isect;
    // The node intersection is performed in the model coordinate system.
    // Therefore, the ray needs to be transformed into the model coordinate system.
    // The intersection attributes returned are all in the model coordinate system.
    // Therefore, They are need to be converted to the world coordinate system.    
    // If the model shrinks, the value of t will also change.
    // The change of t can be solved by intersection point changing simultaneously
    Eigen::Matrix4f model_inv = model.inverse(); 
    Vector3f new_origin = (model_inv * ray.origin.homogeneous()).hnormalized();
    Vector3f new_direction = (model_inv * ray.direction.homogeneous()).hnormalized();
    Ray ray_model(new_origin,new_direction);
    Vector3f inv_dir(1.0f / ray_model.direction.x(), 
                     1.0f / ray_model.direction.y(), 
                     1.0f / ray_model.direction.z());
    std::array<int, 3> dir_is_neg = {
        ray_model.direction.x() < 0,
        ray_model.direction.y() < 0,
        ray_model.direction.z() < 0
    };


    if (!node->aabb.intersect(ray_model,inv_dir,dir_is_neg))
    {
        return std::nullopt;
    }        

    if (node->face_idx != 0) {
        return ray_triangle_intersect(ray_model,mesh,node->face_idx);
    }

    auto left_isect = ray_node_intersect(node->left,ray_model);
    auto right_isect = ray_node_intersect(node->right, ray_model);

    // 合并左右子树的相交结果，返回最近的相交点
    if (left_isect && right_isect) {
        return (left_isect->t < right_isect->t) ? left_isect : right_isect;
    } else if (left_isect) {
        return left_isect;
    } else {
        return right_isect;
    }
    return isect;
}
