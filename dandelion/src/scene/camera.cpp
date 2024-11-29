#include "camera.h"

#include <cmath>
#include <Eigen/Geometry>
#include "../utils/formatter.hpp"
#include <spdlog/spdlog.h>

#include "../utils/math.hpp"

using Eigen::Affine3f;
using Eigen::Matrix3f;
using Eigen::Matrix4f;
using Eigen::Vector3f;
using Eigen::Vector4f;

Camera::Camera(const Eigen::Vector3f& position, const Eigen::Vector3f& target, float near_plane,
               float far_plane, float fov_y_degrees, float aspect_ratio)
    : position(position), target(target), near_plane(near_plane), far_plane(far_plane),
      fov_y_degrees(fov_y_degrees), aspect_ratio(aspect_ratio)
{
    world_up.x() = 0.0f;
    world_up.y() = 1.0f;
    world_up.z() = 0.0f;
}

Matrix4f Camera::view()
{
    Vector3f inv_direction = (position - target).normalized();
    Vector3f right         = (world_up).cross(inv_direction).normalized();
    Vector3f up            = inv_direction.cross(right);
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


    const float top = near_plane * std::tan(radians(fov_y_degrees) / 2);
    const float right = top * aspect_ratio;

    Matrix4f projection;
    projection << near_plane/right, 0, 0, 0,
                  0, near_plane/top, 0, 0,
                  0, 0, -(far_plane + near_plane) / (far_plane - near_plane), -2 * far_plane * near_plane / (far_plane - near_plane),
                  0, 0, -1, 0;

#ifdef DEBUG
    spdlog::trace("top: {}, right: {}", top, right);
    spdlog::trace("near: {}, far: {}", near_plane, far_plane);
    spdlog::trace("Projection matrix: {}\n", projection);
#endif

    return projection;
}
