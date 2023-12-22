#ifndef RAY_H
#define RAY_H

#include "Vec3.hpp"

class Ray {
private:
  Point3 origin;
  Vec3 direction;

public:
  Ray();
  Ray(const Point3 &origin, const Vec3 &direction);

  Point3 getOrigin() const;
  Vec3 getDirection() const;

  Point3 at(double t) const;
};

#endif // RAY_H
