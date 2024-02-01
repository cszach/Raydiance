#include "Scene.cuh"
#include "cuda_helper.cuh"

__device__ Scene::Scene(Object **objects, int num_objects)
    : _objects(objects), _num_objects(num_objects) {}

// __device__ void Scene::add(Object *object) {
//   *(d_objects + _num_objects++) = object;
// }

__device__ bool Scene::hit(const Ray &ray, float t_min, float t_max,
                           HitRecord &rec) const {
  HitRecord temp_record;
  bool hit_anything = false;
  float closest_so_far = t_max;

  for (int i = 0; i < _num_objects; i++) {
    bool got_hit = _objects[i]->hit(ray, t_min, t_max, temp_record);

    if (got_hit && temp_record.t < closest_so_far) {
      hit_anything = true;
      closest_so_far = temp_record.t;
      rec = temp_record;
    }
  }

  return hit_anything;
}
