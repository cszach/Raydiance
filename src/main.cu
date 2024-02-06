#include "BVH.cuh"
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

#define RAND_DOUBLE curand_uniform(localRandState)

__global__ void setup(Object **d_objects, Scene **d_scene, Camera **d_camera,
                      float fov, float aspectRatio, curandState *randState) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curand_init(2024, 0, 0, randState);
    curandState *localRandState = randState;

    int numObjects = 0;
    const float spacing = 1.5;

    for (int x = -3; x < 4; ++x) {
      for (int y = -3; y < 4; ++y) {
        for (int z = -3; z < 4; ++z) {
          float randomDouble = RAND_DOUBLE;
          Material *material;

          if (randomDouble < 0.5) {
            material = new Lambertian(RANDVEC3);
          } else if (randomDouble < 0.8) {
            material = new Metal(RANDVEC3, RAND_DOUBLE);
          } else {
            material = new Dielectric(1.5);
          }

          auto sphere = new Sphere(0.5, material);
          sphere->position.set(x * spacing, y * spacing, z * spacing);

          d_objects[numObjects++] = sphere;
        }
      }
    }

    *d_scene = new Scene(d_objects, numObjects);
    (*d_scene)->computeBoundingBox();
    (*d_scene)->computeBVH(localRandState);

    *d_camera = new Camera(fov, aspectRatio);
    (*d_camera)->position.set(13, 0, 3);
    (*d_camera)->lookAt.set(0, 0, 0);
  }
}

int main() {
  // Image

  const float ASPECT_RATIO = 16.0 / 9.0;
  const float VERTICAL_FOV = 20;
  const int IMAGE_WIDTH = 1200;
  const int IMAGE_HEIGHT = static_cast<int>(IMAGE_WIDTH / ASPECT_RATIO);
  const int NUM_OBJECTS = 7 * 7 * 7;

  std::ofstream f_out("image.ppm");

  cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 4);

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

  curandState *d_worldSetupRandState;
  checkCudaError(
      cudaMalloc((void **)&d_worldSetupRandState, sizeof(curandState)));

  setup<<<1, 1>>>(d_objects, d_scene, d_camera, VERTICAL_FOV, ASPECT_RATIO,
                  d_worldSetupRandState);

  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());

  Renderer renderer(IMAGE_WIDTH, IMAGE_HEIGHT);
  renderer.numSamples = 10;
  renderer.numBounces = 5;

  clock_t start, stop;
  start = clock();

  renderer.render(d_scene, d_camera, d_rand_state);
  checkCudaError(cudaDeviceSynchronize());

  stop = clock();
  double timeSeconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cout << "Rendering took " << timeSeconds << " seconds.\n";

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
