#ifndef RENDERER_H
#define RENDERER_H

#include <curand_kernel.h>

#include "Camera.cuh"
#include "Interval.cuh"
#include "Ray.cuh"
#include "Scene.cuh"
#include "Vec3.cuh"

class DRenderer {
public:
  Camera **_camera;

  Point3 pixel00;
  Vec3 pixelDeltaU;
  Vec3 pixelDeltaV;
  Point3 center;

  // static const Interval intensity;
  int outputWidth;
  int outputHeight;
  int numSamples;
  int numBounces;
  // float *fb;

  __device__ DRenderer(int _outputWidth, int _outputHeight);

  __device__ Ray getRay(int i, int j, curandState *rand_state) const;
  __device__ Point3 getPixelSampleSquare(curandState *rand_state) const;
  __device__ Color getRayColor(const Ray &ray, Scene **scene, int num_bounces,
                               curandState *local_rand_state) const;
};

class Renderer {
public:
  int outputWidth;
  int outputHeight;
  int numSamples = 10;
  int numBounces = 10;

  // TODO: Consider making these private
  DRenderer **d_renderer;
  size_t fb_size;
  float *fb;

  __host__ Renderer(int _outputWidth, int _outputHeight);
  __host__ void render(Scene **d_scene, Camera **d_camera,
                       curandState *d_rand_state);
};

// const Interval Renderer::intensity(0.0, 0.999);

#endif // RENDERER_H