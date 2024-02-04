#include "Material.cuh"
#include "stdio.h"

__device__ Vec3 Material::reflect(const Vec3 &v, const Vec3 &n) {
  return v - 2 * v.dot(n) * n;
}

__device__ Vec3 Material::refract(const Vec3 &uv, const Vec3 &n,
                                  double etaRatio) {
  float cosTheta = fmin(-uv.dot(n), 1.0f);

  Vec3 refractedRayPerpendicular = etaRatio * (uv + cosTheta * n);
  Vec3 refractedRayParallel =
      -sqrt(fabs(1.0f - refractedRayPerpendicular.lengthSquared())) * n;

  return refractedRayPerpendicular + refractedRayParallel;
}

// lambertian

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

// metal

__device__ Metal::Metal(const Color &_albedo, float _fuzziness)
    : albedo(_albedo), fuzziness(_fuzziness) {}

__device__ bool Metal::scatter(const Ray &rayIn, const HitRecord &record,
                               Color &attenuation, Ray &scattered,
                               curandState *localRandState) const {
  Vec3 reflected =
      Material::reflect(rayIn.direction.normalize(), record.normal);
  scattered =
      Ray(record.p, reflected + fuzziness * Vec3::randomUnit(localRandState));
  attenuation = albedo;

  return true;
}

// dielectric

__device__ Dielectric::Dielectric(double _refractionIndex)
    : refractionIndex(_refractionIndex) {}

__device__ bool Dielectric::scatter(const Ray &rayIn, const HitRecord &record,
                                    Color &attenuation, Ray &scattered,
                                    curandState *localRandState) const {
  attenuation = Color(1.0, 1.0, 1.0);
  float refractionRatio =
      record.front_face ? (1.0 / refractionIndex) : refractionIndex;

  Vec3 unitDirection = rayIn.direction.normalize();
  float cosTheta = fmin(-unitDirection.dot(record.normal), 1.0f);
  float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

  bool cannotRefract = refractionRatio * sinTheta > 1.0f;
  Vec3 direction;

  if (cannotRefract || reflectance(cosTheta, refractionRatio) >
                           curand_uniform(localRandState)) { // must reflect
    direction = reflect(unitDirection, record.normal);
  } else {
    direction = refract(unitDirection, record.normal, refractionRatio);
  }

  scattered = Ray(record.p, direction);
  return true;
}
