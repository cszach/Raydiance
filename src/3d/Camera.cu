#include <cmath>

#include "Camera.cuh"
#include "MathUtils.cuh"

__device__ Camera::Camera(float vertical_fov, float aspect_ratio) {
  setVerticalFov(vertical_fov);
  setAspectRatio(aspect_ratio);
}

__device__ float Camera::getVerticalFOV() const { return _vertical_fov; }
__device__ float Camera::getAspectRatio() const { return _aspect_ratio; }
__device__ float Camera::getFocalLength() const { return _focal_length; }
__host__ __device__ Point3 Camera::getViewportU() const { return _viewport_u; }
__host__ __device__ Point3 Camera::getViewportV() const { return _viewport_v; }
__host__ __device__ Point3 Camera::getViewportUpperLeft() const {
  return _viewport_upper_left;
}

__device__ void Camera::setVerticalFov(float vertical_fov) {
  _vertical_fov = vertical_fov;
  _theta = degToRad(vertical_fov);
  _h = tan(_theta / 2);

  setFocalLength(_focal_length);
}

__device__ void Camera::setAspectRatio(float aspect_ratio) {
  _aspect_ratio = aspect_ratio;

  auto viewport_width = _viewport_height * _aspect_ratio;
  _viewport_u = Vec3(viewport_width, 0, 0);

  computerViewportUpperLeft();
}

__device__ void Camera::setFocalLength(float focal_length) {
  _focal_length = focal_length;
  _viewport_height = 2 * _h * _focal_length;

  auto viewport_width = _viewport_height * _aspect_ratio;

  _viewport_u = Vec3(viewport_width, 0, 0);
  _viewport_v = Vec3(0, -_viewport_height, 0);

  computerViewportUpperLeft();
}

__device__ bool Camera::hit(const Ray &ray, float t_min, float t_max,
                            HitRecord &rec) const {
  return false;
}

// PRIVATE

__device__ void Camera::computerViewportUpperLeft() {
  _viewport_upper_left =
      _position - Vec3(0, 0, _focal_length) - _viewport_u / 2 - _viewport_v / 2;
}
