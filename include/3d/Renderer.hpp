#ifndef RENDERER_H
#define RENDERER_H

#include <curand_kernel.h>

#include "Camera.hpp"
#include "Interval.hpp"
#include "Ray.hpp"
#include "Scene.hpp"
#include "Vec3.hpp"

class Renderer {
private:
  Camera **_camera;

  int _num_pixels;
  size_t _frame_buffer_size;

  Point3 _pixel00;
  Vec3 _pixel_delta_u;
  Vec3 _pixel_delta_v;
  Point3 _center;

public:
  // static const Interval intensity;
  int output_width;
  int output_height;
  int num_samples = 10;
  int num_bounces = 10;

  __device__ Renderer(Camera **camera, int output_width, int output_height);

  __device__ Camera **getCamera() const;

  __device__ void setCamera(Camera **camera);

  __device__ Ray getRay(int i, int j, curandState *rand_state) const;
  __device__ Point3 getPixelSampleSquare(curandState *rand_state) const;
  __device__ Color getRayColor(const Ray &ray, Scene **scene, int num_bounces,
                               curandState *local_rand_state) const;
};

// const Interval Renderer::intensity(0.0, 0.999);

__global__ void render(Scene **scene, Renderer **renderer, float *fb,
                       curandState *rand_state);

#endif // RENDERER_H