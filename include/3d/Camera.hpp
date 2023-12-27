#ifndef CAMERA_H
#define CAMERA_H

#include "Object.hpp"
#include "Ray.hpp"
#include "Vec3.hpp"

class Camera : public Object {
private:
  float _vertical_fov;
  float _aspect_ratio;

  float _focal_length = 1.0f;
  Point3 _position = Point3(0, 0, 0);

  float _theta;
  float _h;
  float _viewport_height;
  Vec3 _viewport_u;
  Vec3 _viewport_v;
  Point3 _viewport_upper_left;

public:
  __device__ Camera(float vertical_fov, float aspect_ratio);

  __device__ float getVerticalFOV() const;
  __device__ float getAspectRatio() const;
  __device__ float getFocalLength() const;
  __device__ Vec3 getViewportU() const;
  __device__ Vec3 getViewportV() const;
  __device__ Point3 getViewportUpperLeft() const;

  __device__ void setVerticalFov(float vertical_fov);
  __device__ void setAspectRatio(float aspect_ratio);
  __device__ void setFocalLength(float focal_length);

  __device__ bool hit(const Ray &ray, float t_min, float t_max,
                      HitRecord &rec) const override;

private:
  __device__ void computerViewportUpperLeft();
};

#endif // CAMERA_H
