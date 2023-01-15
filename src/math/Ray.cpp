#include "Ray.hpp"

Ray::Ray() : origin(Point3(0, 0, 0)), direction(Vec3(0, 0, -1)) {}
Ray::Ray(const Point3 &origin, const Vec3 &direction)
    : origin(origin), direction(direction) {}

Point3 Ray::getOrigin() const { return this->origin; }
Vec3 Ray::getDirection() const { return this->direction; }

Point3 Ray::at(double t) const { return this->origin + t * this->direction; }