#include "Material.cuh"
#include "stdio.h"

__device__ Vec3 Material::reflect(const Vec3 &v, const Vec3 &n) {
  return v - 2 * v.dot(n) * n;
}

__device__ Lambertian::Lambertian(const Color &_albedo) : albedo(_albedo) {}

__device__ bool Lambertian::scatter(const Ray &rayIn, const HitRecord &record,
                                    Color &attenuation, Ray &scattered,
                                    curandState *localRandState) const {
  auto scatterDirection = record.normal + Vec3::randomUnit(localRandState);

  if (scatterDirection.isNearZero()) {
    scatterDirection = record.normal;
  }

  scattered = Ray(record.p, scatterDirection);
  attenuation = albedo;

  return true;
}

__device__ Metal::Metal(const Color &_albedo) : albedo(_albedo) {}

__device__ bool Metal::scatter(const Ray &rayIn, const HitRecord &record,
                               Color &attenuation, Ray &scattered,
                               curandState *localRandState) const {
  Vec3 reflected =
      Material::reflect(rayIn.direction.normalize(), record.normal);
  scattered = Ray(record.p, reflected);
  attenuation = albedo;

  return true;
}