#include "Scene.hpp"

Scene::Scene() = default;

void Scene::clear() { objects.clear(); }

void Scene::add(shared_ptr<Object> object) { objects.push_back(object); }

bool Scene::hit(const Ray &ray, double t_min, double t_max,
                HitRecord &rec) const {
  HitRecord temp_record;
  bool hit_anything = false;
  double closest_so_far = t_max;

  for (const auto &object : objects) {
    bool got_hit = object->hit(ray, t_min, t_max, temp_record);

    if (got_hit && temp_record.t < closest_so_far) {
      hit_anything = true;
      closest_so_far = temp_record.t;
      rec = temp_record;
    }
  }

  return hit_anything;
}
