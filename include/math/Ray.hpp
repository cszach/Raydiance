#ifndef RAY_H
#define RAY_H

#include "Vec3.hpp"

class Ray {
private:
  Point3 origin;
  Vec3 direction;

public:
  Ray() : origin(Point3(0, 0, 0)), direction(Vec3(0, 0, -1)) {}
  Ray(const Point3 &origin, const Vec3 &direction)
      : origin(origin), direction(direction) {}

  inline Point3 getOrigin() const { return this->origin; }
  inline Vec3 getDirection() const { return this->direction; }

  Point3 at(double t) const { return this->origin + t * this->direction; }
};

#endif // RAY_H