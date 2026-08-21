#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <cmath>
#include <algorithm>

namespace hearlink {

struct Vec3 {
    float x, y, z;

    Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    Vec3 operator/(float s) const { return Vec3(x / s, y / s, z / s); }

    float length_sq() const { return x * x + y * y + z * z; }
    float length() const { return std::sqrt(length_sq()); }

    Vec3 normalized() const {
        float l = length();
        if (l < 1e-6f) return Vec3(0, 0, 0);
        return *this / l;
    }

    static float dot(const Vec3& a, const Vec3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    static Vec3 cross(const Vec3& a, const Vec3& b) {
        return Vec3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }

    static Vec3 lerp(const Vec3& a, const Vec3& b, float t) {
        return a + (b - a) * t;
    }
};

struct Quaternion {
    float x, y, z, w;

    Quaternion() : x(0.0f), y(0.0f), z(0.0f), w(1.0f) {}
    Quaternion(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}

    float norm_sq() const { return x * x + y * y + z * z + w * w; }
    float norm() const { return std::sqrt(norm_sq()); }

    Quaternion normalized() const {
        float n = norm();
        if (n < 1e-6f) return Quaternion(0, 0, 0, 1);
        return Quaternion(x / n, y / n, z / n, w / n);
    }

    static Quaternion identity() {
        return Quaternion(0.0f, 0.0f, 0.0f, 1.0f);
    }

    static Quaternion slerp(Quaternion q1, Quaternion q2, float t) {
        float dot = q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w;

        if (dot < 0.0f) {
            q2 = Quaternion(-q2.x, -q2.y, -q2.z, -q2.w);
            dot = -dot;
        }

        if (dot > 0.9995f) {
            Quaternion res(
                q1.x + t * (q2.x - q1.x),
                q1.y + t * (q2.y - q1.y),
                q1.z + t * (q2.z - q1.z),
                q1.w + t * (q2.w - q1.w)
            );
            return res.normalized();
        }

        float theta_0 = std::acos(std::min(1.0f, std::max(-1.0f, dot)));
        float theta = theta_0 * t;
        float sin_theta = std::sin(theta);
        float sin_theta_0 = std::sin(theta_0);

        float s1 = std::cos(theta) - dot * sin_theta / sin_theta_0;
        float s2 = sin_theta / sin_theta_0;

        return Quaternion(
            q1.x * s1 + q2.x * s2,
            q1.y * s1 + q2.y * s2,
            q1.z * s1 + q2.z * s2,
            q1.w * s1 + q2.w * s2
        );
    }

    static Quaternion from_to_rotation(const Vec3& from, const Vec3& to) {
        Vec3 v0 = from.normalized();
        Vec3 v1 = to.normalized();
        float d = Vec3::dot(v0, v1);

        if (d >= 0.99999f) {
            return Quaternion::identity();
        }
        if (d <= -0.99999f) {
            Vec3 ortho(1, 0, 0);
            if (std::abs(v0.x) > 0.8f) ortho = Vec3(0, 1, 0);
            Vec3 axis = Vec3::cross(v0, ortho).normalized();
            return Quaternion(axis.x, axis.y, axis.z, 0.0f);
        }

        Vec3 axis = Vec3::cross(v0, v1);
        float s = std::sqrt((1.0f + d) * 2.0f);
        float inv_s = 1.0f / s;

        return Quaternion(
            axis.x * inv_s,
            axis.y * inv_s,
            axis.z * inv_s,
            s * 0.5f
        ).normalized();
    }
};

} // namespace hearlink

#endif // MATH_UTILS_H
#endif
