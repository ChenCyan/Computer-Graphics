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
