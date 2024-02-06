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

    auto floor = new Sphere(1000, new Lambertian(Color(0.5, 0.5, 0.5)));
    floor->position.set(0, -1000, 0);
    *(d_objects + numObjects++) = floor;

    for (int a = -11; a < 11; a++) {
      for (int b = -11; b < 11; b++) {
        float randomDouble = RAND_DOUBLE;

        Point3 center(a + 0.9 * randomDouble, 0.2, b + 0.9 * randomDouble);

        if ((center - Point3(4, 0.2, 0)).length() > 0.9) {
          Sphere *sphere;

          if (randomDouble < 0.8) { // diffuse
            Color albedo = RANDVEC3 * RANDVEC3;

            sphere = new Sphere(0.2, new Lambertian(albedo));
            sphere->position = center;
            *(d_objects + numObjects++) = sphere;
          } else if (randomDouble < 0.95) { // metal
            Color albedo =
                Vec3(0.5f * (1.0f + RAND_DOUBLE), 0.5f * (1.0f + RAND_DOUBLE),
                     0.5f * (1.0f + RAND_DOUBLE));
            float fuzziness = 0.5f * RAND_DOUBLE;

            sphere = new Sphere(0.2, new Metal(albedo, fuzziness));
            sphere->position = center;
            *(d_objects + numObjects++) = sphere;
          } else { // glass
            sphere = new Sphere(0.2, new Dielectric(1.5));
            sphere->position = center;
            *(d_objects + numObjects++) = sphere;
          }
        }
      }
    }

    Sphere *bigSphere1 = new Sphere(1.0, new Dielectric(1.5));
    bigSphere1->position.set(0, 1, 0);

    Sphere *bigSphere2 = new Sphere(1.0, new Lambertian(Color(0.4, 0.2, 0.1)));
    bigSphere2->position.set(-4, 1, 0);

    Sphere *bigSphere3 = new Sphere(1.0, new Metal(Color(0.7, 0.6, 0.5), 0.0));
    bigSphere3->position.set(4, 1, 0);

    *(d_objects + numObjects++) = bigSphere1;
    *(d_objects + numObjects++) = bigSphere2;
    *(d_objects + numObjects++) = bigSphere3;

    *d_scene = new Scene(d_objects, numObjects);
    (*d_scene)->computeBoundingBox();
    (*d_scene)->computeBVH(localRandState);

    *d_camera = new Camera(fov, aspectRatio);
    (*d_camera)->position.set(13, 2, 3);
    (*d_camera)->lookAt.set(0, 0, 0);
  }
}

int main() {
  // Image

  const float ASPECT_RATIO = 16.0 / 9.0;
  const float VERTICAL_FOV = 20;
  const int IMAGE_WIDTH = 1200;
  const int IMAGE_HEIGHT = static_cast<int>(IMAGE_WIDTH / ASPECT_RATIO);
  const int NUM_OBJECTS = 22 * 22 + 4;

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
  renderer.numSamples = 5;
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
