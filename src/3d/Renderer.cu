#include <curand_kernel.h>
#include <iostream>

#include "MathUtils.hpp"
#include "Renderer.hpp"

__device__ Renderer::Renderer(Camera **camera, int output_width,
                              int output_height)
    : _camera(camera), output_width(output_width),
      output_height(output_height) {
  setCamera(camera);
}

__device__ Camera **Renderer::getCamera() const { return _camera; }

__device__ void Renderer::setCamera(Camera **camera) {
  _camera = camera;

  auto viewport_u = (*camera)->getViewportU();
  auto viewport_v = (*camera)->getViewportV();

  _pixel_delta_u = viewport_u / output_width;
  _pixel_delta_v = viewport_v / output_height;

  _pixel00 = (*camera)->getViewportUpperLeft() +
             0.5 * (_pixel_delta_u + _pixel_delta_v);

  _center = (*camera)->getPosition();
}

__global__ void render(Object **scene, Renderer **renderer, float *fb,
                       curandState *rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= (*renderer)->output_width || j >= (*renderer)->output_height) {
    return;
  }

  int pixel_index = 3 * (i + j * (*renderer)->output_width);

  curand_init(2023, pixel_index / 3, 0, &rand_state[pixel_index / 3]);
  curandState local_rand_state = rand_state[pixel_index / 3];

  Color pixel_color(0, 0, 0);

  for (int sample = 0; sample < (*renderer)->num_samples; ++sample) {
    Ray ray = (*renderer)->getRay(i, j, &local_rand_state);
    pixel_color += (*renderer)->getRayColor(
        ray, scene, (*renderer)->num_bounces, &local_rand_state);
  }

  // Process samples

  pixel_color /= (*renderer)->num_samples;

  auto r = pixel_color.x;
  auto g = pixel_color.y;
  auto b = pixel_color.z;

  // Apply linear to gamma transform

  r = pow(r, 0.4545f);
  g = pow(g, 0.4545f);
  b = pow(b, 0.4545f);

  //  Write color

  fb[pixel_index] = r < 0.0 ? 0.0 : (r > 0.99 ? 0.99 : r);
  fb[pixel_index + 1] = g < 0.0 ? 0.0 : (g > 0.99 ? 0.99 : g);
  fb[pixel_index + 2] = b < 0.0 ? 0.0 : (b > 0.99 ? 0.99 : b);
}

// PRIVATE (?)

__device__ Ray Renderer::getRay(int i, int j,
                                curandState *local_rand_state) const {
  auto pixel_center = _pixel00 + i * _pixel_delta_u + j * _pixel_delta_v;
  auto pixel_sample = pixel_center + getPixelSampleSquare(local_rand_state);

  return Ray(_center, pixel_sample - _center);
}

__device__ Point3
Renderer::getPixelSampleSquare(curandState *local_rand_state) const {
  auto x = -0.5 + random_float(local_rand_state);
  auto y = -0.5 + random_float(local_rand_state);

  return x * _pixel_delta_u + y * _pixel_delta_v;
}

__device__ Color Renderer::getRayColor(const Ray &ray, Object **scene,
                                       int num_bounces,
                                       curandState *local_rand_state) const {
  Ray r = ray;
  float attenuation = 1.0f;

  for (int i = 0; i < num_bounces; ++i) {
    HitRecord rec;

    if ((*scene)->hit(r, 0.001, INFINITY, rec)) {
      Vec3 direction = rec.normal + Vec3::randomUnit(local_rand_state);

      attenuation *= 0.5;
      r = Ray(rec.p, direction);
    } else {
      Vec3 unit_direction = r.direction.normalize();
      auto a = 0.5 * (unit_direction.y + 1.0);
      Color color = (1.0 - a) * Color(1, 1, 1) + a * Color(0.5, 0.7, 1.0);

      return color * attenuation;
    }
  }

  return Color();
}
