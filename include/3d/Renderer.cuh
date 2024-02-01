#ifndef RENDERER_H
#define RENDERER_H

#include <curand_kernel.h>

#include "Camera.cuh"
#include "Interval.cuh"
#include "Ray.cuh"
#include "Scene.cuh"
#include "Vec3.cuh"

class DRenderer {
private:
  Camera **_camera;

  int _num_pixels;
  size_t _fb_size;

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
  // float *fb;

  __device__ DRenderer(Camera **camera, int output_width, int output_height);

  __device__ Camera **getCamera() const;

  __device__ void setCamera(Camera **camera);

  __device__ Ray getRay(int i, int j, curandState *rand_state) const;
  __device__ Point3 getPixelSampleSquare(curandState *rand_state) const;
  __device__ Color getRayColor(const Ray &ray, Scene **scene, int num_bounces,
                               curandState *local_rand_state) const;
};

class Renderer {
public:
  int output_width;
  int output_height;

  // TODO: Consider making these private
  DRenderer **d_renderer;
  size_t fb_size;
  float *fb;

  __host__ Renderer(Camera **d_camera, int output_width, int output_height);
  __host__ void render(Scene **scene, curandState *d_rand_state);
};

// const Interval Renderer::intensity(0.0, 0.999);

#endif // RENDERER_H