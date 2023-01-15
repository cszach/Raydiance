#ifndef VEC3_H
#define VEC3_H

#include <cmath>

class Vec3 {
private:
  double x;
  double y;
  double z;

public:
  Vec3() : x(0), y(0), z(0) {}
  Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

  double getX() const { return this->x; }
  double getY() const { return this->y; }
  double getZ() const { return this->z; }

  void setX(double x) { this->x = x; }
  void setY(double y) { this->y = y; }
  void setZ(double z) { this->z = z; }

  double length() const { return sqrt(this->lengthSquared()); }

  double lengthSquared() const {
    return this->x * this->x + this->y * this->y + this->z * this->z;
  }

  bool equals(const Vec3 &v) const {
    return this->x == v.getX() && this->y == v.getY() && this->z == v.getZ();
  }

  Vec3 operator-() const { return Vec3(-this->x, -this->y, -this->z); }

  Vec3 &operator+=(const Vec3 &v) {
    this->x += v.getX();
    this->y += v.getY();
    this->z += v.getZ();

    return *this;
  }

  Vec3 &operator-=(const Vec3 &v) {
    this->x -= v.getX();
    this->y -= v.getY();
    this->z -= v.getZ();

    return *this;
  }

  Vec3 &operator*=(const Vec3 &v) {
    this->x *= v.getX();
    this->y *= v.getY();
    this->z *= v.getZ();

    return *this;
  }

  Vec3 &operator/=(const double t) {
    this->x /= t;
    this->y /= t;
    this->z /= t;

    return *this;
  }
};

inline Vec3 operator+(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.getX() + v.getX(), u.getY() + v.getY(), u.getZ() + v.getZ());
}

inline Vec3 operator-(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.getX() - v.getX(), u.getY() - v.getY(), u.getZ() - v.getZ());
}

inline Vec3 operator*(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.getX() * v.getX(), u.getY() * v.getY(), u.getZ() * v.getZ());
}

inline Vec3 operator*(const Vec3 &v, const double t) {
  return Vec3(v.getX() * t, v.getY() * t, v.getZ() * t);
}

inline Vec3 operator*(const double t, const Vec3 &v) { return v * t; }

inline Vec3 operator/(const Vec3 &v, const double t) { return v * (1 / t); }

inline double dotProduct(const Vec3 &u, const Vec3 &v) {
  return u.getX() * v.getX() + u.getY() * v.getY() + u.getZ() * v.getZ();
}

inline Vec3 crossProduct(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.getY() * v.getZ() - u.getZ() * v.getY(),
              u.getZ() * v.getX() - u.getX() * v.getZ(),
              u.getX() * v.getY() - u.getY() * v.getX());
}

inline Vec3 unitVectorFrom(const Vec3 &v) { return v / v.length(); }

using Point3 = Vec3;
using Color = Vec3;

#endif // VEC3_H