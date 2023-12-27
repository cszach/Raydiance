#ifndef SCENE_H
#define SCENE_H

#include <memory>
#include <vector>

#include "Object.hpp"

class Scene : public Object {
public:
  Object **_objects;
  int _num_objects = 0;

  __device__ Scene(Object **objects, int num_objects);

  // __device__ void add(Object *object);

  __device__ bool hit(const Ray &ray, float t_min, float t_max,
                      HitRecord &rec) const override;
};

#endif // SCENE_H
