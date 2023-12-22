#ifndef VEC3_H
#define VEC3_H

#include <cmath>

class Vec3 {
private:
  double _x;
  double _y;
  double _z;

public:
  Vec3();
  Vec3(double x, double y, double z);

  double getX() const;
  double getY() const;
  double getZ() const;

  void setX(double x);
  void setY(double y);
  void setZ(double z);

  double length() const;

  double lengthSquared() const;

  bool equals(const Vec3 &v) const;

  Vec3 operator-() const;

  Vec3 &operator+=(const Vec3 &v);

  Vec3 &operator-=(const Vec3 &v);

  Vec3 &operator*=(const Vec3 &v);

  Vec3 &operator/=(const double t);
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
