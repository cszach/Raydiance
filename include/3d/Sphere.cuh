#ifndef SPHERE_H
#define SPHERE_H

#include "Object.cuh"
#include "Vec3.cuh"

class Sphere : public Object {
private:
  float _radius;

public:
  __device__ explicit Sphere(float radius);

  __device__ bool hit(const Ray &ray, float t_min, float t_max,
                      HitRecord &rec) const override;
};

#endif // SPHERE_H
