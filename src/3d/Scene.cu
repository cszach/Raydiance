#include "Scene.cuh"
#include "cuda_helper.cuh"
#include <cstdio>

__device__ Scene::Scene(Object **_objects, int _numobjects)
    : objects(_objects), count(_numobjects) {}

__device__ void Scene::computeBVH(curandState *localRandState) {
  bvh = new BVHNode(objects, 0, count, localRandState);
}

// __device__ void Scene::add(Object *object) {
//   *(dobjects + count++) = object;
// }

__device__ bool Scene::hit(const Ray &ray, Interval ray_t,
                           HitRecord &rec) const {
  HitRecord record;
  bool hitAnything = false;
  float closest = ray_t.max;

  // Brute force
  // for (int i = 0; i < count; i++) {
  //   bool got_hit = objects[i]->hit(ray, t_min, t_max, record);

  //   if (got_hit && record.t < closest) {
  //     hitAnything = true;
  //     closest = record.t;
  //     rec = record;
  //   }
  // }

  // BVH
  bool gotHit = bvh->hit(ray, Interval(ray_t.min, closest), record);

  if (gotHit && record.t < closest) {
    hitAnything = true;
    closest = record.t;
    rec = record;
  }

  return hitAnything;
}

__device__ void Scene::computeBoundingBox() {
  boundingBox = AABB();

  for (int i = 0; i < count; i++) {
    (*(objects + i))->computeBoundingBox();
    boundingBox = AABB(boundingBox, objects[i]->boundingBox);
  }
}
