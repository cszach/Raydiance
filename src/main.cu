#include <curand_kernel.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdio.h>

#include "Renderer.hpp"
#include "Scene.hpp"
#include "Sphere.hpp"
#include "cuda_helper.hpp"

__global__ void setup(Object **d_objects, Scene **d_scene, Camera **d_camera,
                      float vertical_fov, float aspect_ratio) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    auto test_sphere = new Sphere(0.5);
    test_sphere->setPosition(Point3(0, 0, -1));

    auto floor = new Sphere(100);
    floor->setPosition(Point3(0, -100.5, -1));

    *(d_objects) = test_sphere;
    *(d_objects + 1) = floor;

    *d_scene = new Scene(d_objects, 2);

    *d_camera = new Camera(vertical_fov, aspect_ratio);
  }
}

int main() {
  // Image

  const float ASPECT_RATIO = 2.0;
  const float VERTICAL_FOV = 100;
  const int IMAGE_WIDTH = 1200;
  const int IMAGE_HEIGHT = static_cast<int>(IMAGE_WIDTH / ASPECT_RATIO);
  const int NUM_OBJECTS = 2;

  std::ofstream f_out("image.ppm");

  // Camera

  Camera **d_camera;
  checkCudaError(cudaMalloc((void **)&d_camera, sizeof(Camera *)));

  // Scene

  Object **d_objects;
  checkCudaError(
      cudaMalloc((void **)&d_objects, NUM_OBJECTS * sizeof(Object *)));

  Scene **d_scene;
  checkCudaError(cudaMalloc((void **)&d_scene, sizeof(Scene *)));

  // Render
  curandState *d_rand_state;
  checkCudaError(cudaMalloc((void **)&d_rand_state,
                            IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(curandState)));

  setup<<<1, 1>>>(d_objects, d_scene, d_camera, VERTICAL_FOV, ASPECT_RATIO);

  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());

  Renderer renderer(d_camera, IMAGE_WIDTH, IMAGE_HEIGHT);

  renderer.render(d_scene, d_rand_state);

  // Write to PPM

  f_out << "P3\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << '\n' << 255 << '\n';

  for (int j = 0; j < IMAGE_HEIGHT; ++j) {
    for (int i = 0; i < IMAGE_WIDTH; ++i) {
      int pixel_index = 3 * (i + j * IMAGE_WIDTH);

      float r = renderer.fb[pixel_index + 0];
      float g = renderer.fb[pixel_index + 1];
      float b = renderer.fb[pixel_index + 2];

      auto ir = int(255.99 * r);
      auto ig = int(255.99 * g);
      auto ib = int(255.99 * b);

      f_out << ir << ' ' << ig << ' ' << ib << '\n';
    }
  }

  checkCudaError(cudaDeviceReset());
}
