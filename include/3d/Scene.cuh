#ifndef SCENE_H
#define SCENE_H

#include "BVH.cuh"
#include "Object.cuh"
#include <memory>
#include <vector>

class Scene : public Object {
private:
  Object **objects;
  int count = 0;
  BVHNode *bvh;

public:
  __device__ Scene(Object **_objects, int _numobjects);

  __device__ void computeBVH(curandState *localRandState);

  // __device__ void add(Object *object);

  __device__ bool hit(const Ray &ray, Interval ray_t,
                      HitRecord &rec) const override;

  __device__ virtual void computeBoundingBox() override;
};

#endif // SCENE_H
