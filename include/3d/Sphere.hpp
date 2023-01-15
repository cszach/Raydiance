#ifndef SPHERE_H
#define SPHERE_H

#include "Object.hpp"
#include "Vec3.hpp"

class Sphere : public Object {
private:
  double radius;

public:
  Sphere(double radius);

  bool hit(const Ray &ray, double t_min, double t_max,
           HitRecord &rec) const override;
};

#endif // SPHERE_H