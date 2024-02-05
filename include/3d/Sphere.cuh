#ifndef SPHERE_H
#define SPHERE_H

#include "Object.cuh"
#include "Vec3.cuh"

class Sphere : public Object {
private:
  float radius;

public:
  __device__ Sphere(float _radius, Material *_material);

  __device__ bool hit(const Ray &ray, float t_min, float t_max,
                      HitRecord &rec) const override;

  __device__ virtual void computeBoundingBox() override;
};

#endif // SPHERE_H
