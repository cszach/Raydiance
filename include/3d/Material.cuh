#ifndef MATERIAL_H
#define MATERIAL_H

struct HitRecord;

#include "Object.cuh"
#include "Ray.cuh"
#include "Vec3.cuh"

class Material {
public:
  virtual ~Material() = default;

  __device__ virtual bool scatter(const Ray &rayIn, const HitRecord &record,
                                  Color &attenuation, Ray &scattered,
                                  curandState *localRandState) const = 0;

  __device__ static Vec3 reflect(const Vec3 &v, const Vec3 &n);
  __device__ static Vec3 refract(const Vec3 &uv, const Vec3 &n,
                                 double etaRatio);
};

class Lambertian : public Material {
private:
  Color albedo;

public:
  __device__ Lambertian(const Color &_albedo);

  __device__ bool scatter(const Ray &rayIn, const HitRecord &record,
                          Color &attenuation, Ray &scattered,
                          curandState *localRandState) const override;
};

class Metal : public Material {
private:
  Color albedo;
  float fuzziness;

public:
  __device__ Metal(const Color &_albedo, float _fuzziness);

  __device__ bool scatter(const Ray &rayIn, const HitRecord &record,
                          Color &attenuation, Ray &scattered,
                          curandState *localRandState) const override;
};

class Dielectric : public Material {
private:
  float refractionIndex;

  __device__ static float reflectance(float cosine, float refractionIndex) {
    // Schlick's approximation
    float r0 = (1.0f - refractionIndex) / (1.0f + refractionIndex);
    r0 = r0 * r0;

    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5);
  }

public:
  __device__ Dielectric(double _refractionIndex);

  __device__ bool scatter(const Ray &rayIn, const HitRecord &record,
                          Color &attenuation, Ray &scattered,
                          curandState *localRandState) const override;
};

#endif // MATERIAL_H