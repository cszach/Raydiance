#include "Material.cuh"
#include "Renderer.cuh"
#include "Scene.cuh"
#include "Sphere.cuh"
#include "cuda_helper.cuh"
#include <curand_kernel.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdio.h>

__global__ void setup(Object **d_objects, Scene **d_scene, Camera **d_camera,
                      float fov, float aspectRatio) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    auto floor = new Sphere(100, new Lambertian(Color(0.8, 0.8, 0.0)));
    floor->position = Point3(0, -100.5, -1);

    auto center_sphere = new Sphere(0.5, new Lambertian(Color(0.1, 0.2, 0.5)));
    center_sphere->position = Point3(0, 0, -1);

    auto left_sphere = new Sphere(0.5, new Dielectric(1.5));
    left_sphere->position = Point3(-1, 0, -1);

    auto right_sphere = new Sphere(0.5, new Metal(Color(0.8, 0.6, 0.2), 0.0));
    right_sphere->position = Point3(1, 0, -1);

    *(d_objects) = floor;
    *(d_objects + 1) = center_sphere;
    *(d_objects + 2) = left_sphere;
    *(d_objects + 3) = right_sphere;

    *d_scene = new Scene(d_objects, 4);

    *d_camera = new Camera(fov, aspectRatio);
    (*d_camera)->position = Point3(-2, 2, 1);
    (*d_camera)->lookAt = Point3(0, 0, -1);
  }
}

int main() {
  // Image

  const float ASPECT_RATIO = 16.0 / 9.0;
  const float VERTICAL_FOV = 20;
  const int IMAGE_WIDTH = 400;
  const int IMAGE_HEIGHT = static_cast<int>(IMAGE_WIDTH / ASPECT_RATIO);
  const int NUM_OBJECTS = 4;

  std::ofstream f_out("image.ppm");

  cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 2);

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

  Renderer renderer(IMAGE_WIDTH, IMAGE_HEIGHT);

  renderer.render(d_scene, d_camera, d_rand_state);

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
