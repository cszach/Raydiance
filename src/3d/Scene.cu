#include "Scene.cuh"
#include "cuda_helper.cuh"

__device__ Scene::Scene(Object **objects, int numobjects)
    : objects(objects), count(numobjects) {}

// __device__ void Scene::add(Object *object) {
//   *(dobjects + count++) = object;
// }

__device__ bool Scene::hit(const Ray &ray, float t_min, float t_max,
                           HitRecord &rec) const {
  HitRecord temp_record;
  bool hit_anything = false;
  float closest_so_far = t_max;

  for (int i = 0; i < count; i++) {
    bool got_hit = objects[i]->hit(ray, t_min, t_max, temp_record);

    if (got_hit && temp_record.t < closest_so_far) {
      hit_anything = true;
      closest_so_far = temp_record.t;
      rec = temp_record;
    }
  }

  return hit_anything;
}

__device__ void Scene::computeBoundingBox() {
  boundingBox = AABB();

  for (int i = 0; i < count; i++) {
    objects[i]->computeBoundingBox();
    boundingBox = AABB(boundingBox, objects[i]->boundingBox);
  }
}
