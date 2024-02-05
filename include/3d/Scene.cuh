#ifndef SCENE_H
#define SCENE_H

#include <memory>
#include <vector>

#include "Object.cuh"

class Scene : public Object {
public:
  Object **objects;
  int count = 0;

  __device__ Scene(Object **objects, int numobjects);

  // __device__ void add(Object *object);

  __device__ bool hit(const Ray &ray, float t_min, float t_max,
                      HitRecord &rec) const override;

  __device__ virtual void computeBoundingBox() override;
};

#endif // SCENE_H
