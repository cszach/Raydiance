#ifndef BVH_H
#define BVH_H

#include "Object.cuh"

class BVHNode : public Object {
private:
  Object *left;
  Object *right;

  __device__ static void sortPrimitives(Object **objects, int start, int end,
                                        bool(comparator)(const Object *a,
                                                         const Object *b));
  __device__ static bool boxCompare(const Object *a, const Object *b, int axis);
  __device__ static bool boxXCompare(const Object *a, const Object *b);
  __device__ static bool boxYCompare(const Object *a, const Object *b);
  __device__ static bool boxZCompare(const Object *a, const Object *b);

public:
  __device__ BVHNode(Object **objects, size_t start, size_t end,
                     curandState *randState);

  __device__ bool hit(const Ray &ray, Interval ray_t,
                      HitRecord &rec) const override;

  __device__ virtual void computeBoundingBox() override;
};

#endif // BVH_H