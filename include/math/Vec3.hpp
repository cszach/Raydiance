#ifndef VEC3_H
#define VEC3_H

#include <cmath>

#include "MathUtils.hpp"

class Vec3 {
public:
  double x;
  double y;
  double z;

  Vec3();
  Vec3(double x, double y, double z);

  void set(double x, double y, double z);

  double length() const;
  double lengthSquared() const;
  bool equals(const Vec3 &v) const;

  static Vec3 random() {
    return Vec3(random_double(), random_double(), random_double());
  }

  static Vec3 random(double min, double max) {
    return Vec3(random_double(min, max), random_double(min, max),
                random_double(min, max));
  }

  Vec3 operator-() const;
  Vec3 &operator+=(const Vec3 &v);
  Vec3 &operator-=(const Vec3 &v);
  Vec3 &operator*=(const Vec3 &v);
  Vec3 &operator/=(const double t);

  Vec3 normalize() const;
  double dot(const Vec3 &v) const;
  Vec3 cross(const Vec3 &v) const;

  static Vec3 randomInUnitSphere();
  static Vec3 randomUnit();
  static Vec3 randomOnHemisphere(const Vec3 &normal);
};

inline Vec3 operator+(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.x + v.x, u.y + v.y, u.z + v.z);
}

inline Vec3 operator-(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.x - v.x, u.y - v.y, u.z - v.z);
}

inline Vec3 operator*(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.x * v.x, u.y * v.y, u.z * v.z);
}

inline Vec3 operator*(const Vec3 &v, const double t) {
  return Vec3(v.x * t, v.y * t, v.z * t);
}

inline Vec3 operator*(const double t, const Vec3 &v) { return v * t; }

inline Vec3 operator/(const Vec3 &v, const double t) { return v * (1 / t); }

inline Vec3 Vec3::normalize() const {
  double l = length();

  return Vec3(x / l, y / l, z / l);
}

inline double Vec3::dot(const Vec3 &v) const {
  return x * v.x + y * v.y + z * v.z;
}

inline Vec3 Vec3::cross(const Vec3 &v) const {
  return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
}

inline Vec3 Vec3::randomInUnitSphere() {
  while (true) {
    auto v = Vec3::random(-1, 1);

    if (v.lengthSquared() < 1) {
      return v;
    }
  }
}

inline Vec3 Vec3::randomUnit() { return randomInUnitSphere().normalize(); }

inline Vec3 Vec3::randomOnHemisphere(const Vec3 &normal) {
  Vec3 v = randomUnit();

  return v.dot(normal) > 0 ? v : -v;
}

using Point3 = Vec3;
using Color = Vec3;

#endif // VEC3_H
