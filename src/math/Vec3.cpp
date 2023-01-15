#include "Vec3.hpp"

#include <cmath>

Vec3::Vec3() : x(0), y(0), z(0) {}
Vec3::Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

double Vec3::getX() const { return this->x; }
double Vec3::getY() const { return this->y; }
double Vec3::getZ() const { return this->z; }

void Vec3::setX(double x) { this->x = x; }
void Vec3::setY(double y) { this->y = y; }
void Vec3::setZ(double z) { this->z = z; }

double Vec3::length() const { return sqrt(this->lengthSquared()); }

double Vec3::lengthSquared() const {
  return this->x * this->x + this->y * this->y + this->z * this->z;
}

bool Vec3::equals(const Vec3 &v) const {
  return this->x == v.getX() && this->y == v.getY() && this->z == v.getZ();
}

Vec3 Vec3::operator-() const { return Vec3(-this->x, -this->y, -this->z); }

Vec3 &Vec3::operator+=(const Vec3 &v) {
  this->x += v.getX();
  this->y += v.getY();
  this->z += v.getZ();

  return *this;
}

Vec3 &Vec3::operator-=(const Vec3 &v) {
  this->x -= v.getX();
  this->y -= v.getY();
  this->z -= v.getZ();

  return *this;
}

Vec3 &Vec3::operator*=(const Vec3 &v) {
  this->x *= v.getX();
  this->y *= v.getY();
  this->z *= v.getZ();

  return *this;
}

Vec3 &Vec3::operator/=(const double t) {
  this->x /= t;
  this->y /= t;
  this->z /= t;

  return *this;
}