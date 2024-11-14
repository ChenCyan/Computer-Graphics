#include "camera.h"

#include <cmath>
#include <Eigen/Geometry>
#include "../utils/formatter.hpp"
#include <spdlog/spdlog.h>

#include "../utils/math.hpp"

#ifdef _WIN32
#undef near
#undef far
#endif

using Eigen::Affine3f;
using Eigen::Matrix3f;
using Eigen::Matrix4f;
using Eigen::Vector3f;
using Eigen::Vector4f;

Camera::Camera(const Eigen::Vector3f& position, const Eigen::Vector3f& target, float near_plane,
               float far_plane, float fov_y_degrees, float aspect_ratio)
    : position(position), target(target), near(near_plane), far(far_plane),
      fov_y_degrees(fov_y_degrees), aspect_ratio(aspect_ratio)
{
    world_up.x() = 0.0f;
    world_up.y() = 1.0f;
    world_up.z() = 0.0f;
}

Matrix4f Camera::view()
{
    // Compute the inverted view direction, up and right vectors related to
    // the camera itself.
    Vector3f inv_direction = (position - target).normalized();
    Vector3f right         = (world_up).cross(inv_direction).normalized();
    Vector3f up            = inv_direction.cross(right);
    // The view matrix is multiplication of a rotation and a translation
    // matrices: rotation represented by [right, up, inv_direction]^T and
    // translation represented by -position.
    Matrix4f view_matrix          = Matrix4f::Identity();
    view_matrix.block<1, 3>(0, 0) = right;
    view_matrix.block<1, 3>(1, 0) = up;
    view_matrix.block<1, 3>(2, 0) = inv_direction;
    view_matrix(0, 3)             = -right.dot(position);
    view_matrix(1, 3)             = -up.dot(position);
    view_matrix(2, 3)             = -inv_direction.dot(position);
    return view_matrix;
}

Matrix4f Camera::projection()
{
    const float fov_y = radians(fov_y_degrees);
    const float top   = near* std::tan (fov_y / 2);
    const float right = top * aspect_ratio;

    Matrix4f projection = Matrix4f::Zero();
    projection(0, 0) = 1.0f / right;
    projection(1, 1) = 1.0f / top;
    projection(2, 2) = -1.0f / (far - near);
    projection(2, 3) = 0;
    projection(3, 3) = 1.0f;

    Matrix4f persp2ortho = Matrix4f::Zero();
    persp2ortho(0,0) = near;
    persp2ortho(1,1) = near;
    persp2ortho(3,2) = -1;
    persp2ortho(2,2) = (near + far);
    persp2ortho(2,3) = 2 * (near * far);


    Matrix4f projection_m = projection * persp2ortho;

    return projection_m;
}
