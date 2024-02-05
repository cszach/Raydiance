#ifndef CAMERA_H
#define CAMERA_H

#include "Object.cuh"
#include "Ray.cuh"
#include "Vec3.cuh"

class Camera : public Object {
public:
  float fov = 90.0f;
  float aspectRatio = 1.0f;

  float focalLength = 1.0f;
  Point3 position = Point3(0, 0, 0);

  Point3 lookAt = Point3(0, 0, 0);
  Vec3 up = Vec3(0, 1, 0);

  __device__ Camera(float vertical_fov, float aspect_ratio);

  __device__ bool hit(const Ray &ray, float tMin, float tMax,
                      HitRecord &rec) const override;

  __device__ virtual void computeBoundingBox() override;
};

#endif // CAMERA_H
