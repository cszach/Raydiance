#include <cmath>

#include "Camera.cuh"

__device__ Camera::Camera(float _fov, float _aspectRatio)
    : fov(_fov), aspectRatio(_aspectRatio) {}

__device__ bool Camera::hit(const Ray &ray, float tMin, float tMax,
                            HitRecord &rec) const {
  return false;
}

__device__ void Camera::computeBoundingBox() { return; }