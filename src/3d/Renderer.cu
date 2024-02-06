#include <curand_kernel.h>
#include <iostream>

#include "Material.cuh"
#include "MathUtils.cuh"
#include "Renderer.cuh"
#include "Scene.cuh"
#include "cuda_helper.cuh"

// global

__global__ void setupRenderer(DRenderer **d_renderer, int outputWidth,
                              int outputHeight) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *d_renderer = new DRenderer(outputWidth, outputHeight);
  }
}

__global__ void preRender(DRenderer **d_renderer, Camera **d_camera,
                          int numSamples, int numBounces,
                          curandState *rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Initialize camera

    const Camera *camera = *d_camera;
    float focalLength = (camera->position - camera->lookAt).length();
    float fovRadians = degToRad(camera->fov);
    float h = tan(fovRadians / 2.0f);
    float viewportHeight = 2.0f * h * focalLength;
    float viewportWidth = viewportHeight * camera->aspectRatio;

    Vec3 w = (camera->position - camera->lookAt)
                 .normalize();                // Opposite of camera direction
    Vec3 u = camera->up.cross(w).normalize(); // Local right
    Vec3 v = w.cross(u);                      // Local up

    Vec3 viewportU = u * viewportWidth;
    Vec3 viewportV = -v * viewportHeight;

    Vec3 pixelDeltaU = viewportU / (*d_renderer)->outputWidth;
    Vec3 pixelDeltaV = viewportV / (*d_renderer)->outputHeight;

    Point3 center = camera->position;

    Point3 viewportUpperLeft =
        center - (focalLength * w) - viewportU / 2 - viewportV / 2;
    Point3 pixel00 = viewportUpperLeft + 0.5 * (pixelDeltaU + pixelDeltaV);

    (*d_renderer)->pixel00 = pixel00;
    (*d_renderer)->pixelDeltaU = pixelDeltaU;
    (*d_renderer)->pixelDeltaV = pixelDeltaV;
    (*d_renderer)->center = center;
    (*d_renderer)->numSamples = numSamples;
    (*d_renderer)->numBounces = numBounces;
  }
}

__global__ void p_render(Scene **scene, DRenderer **d_renderer, float *fb,
                         curandState *rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= (*d_renderer)->outputWidth || j >= (*d_renderer)->outputHeight) {
    return;
  }

  int pixel_index = 3 * (i + j * (*d_renderer)->outputWidth);

  curand_init(2024, pixel_index / 3, 0, &rand_state[pixel_index / 3]);
  curandState local_rand_state = rand_state[pixel_index / 3];

  Color pixel_color(0, 0, 0);

  for (int sample = 0; sample < (*d_renderer)->numSamples; ++sample) {
    Ray ray = (*d_renderer)->getRay(i, j, &local_rand_state);
    pixel_color += (*d_renderer)
                       ->getRayColor(ray, scene, (*d_renderer)->numBounces,
                                     &local_rand_state);
  }

  // Process samples

  pixel_color /= (*d_renderer)->numSamples;

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

__host__ Renderer::Renderer(int _outputWidth, int _outputHeight)
    : outputWidth(_outputWidth), outputHeight(_outputHeight),
      fb_size(_outputWidth * _outputHeight * 3 * sizeof(float)) {
  checkCudaError(cudaMallocManaged((void **)&fb, fb_size));

  checkCudaError(cudaMalloc((void **)&d_renderer, sizeof(DRenderer *)));

  setupRenderer<<<1, 1>>>(d_renderer, outputWidth, outputHeight);

  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
}

__device__ DRenderer::DRenderer(int _outputWidth, int _outputHeight)
    : outputWidth(_outputWidth), outputHeight(_outputHeight) {}

__host__ void Renderer::render(Scene **d_scene, Camera **d_camera,
                               curandState *d_rand_state) {
  preRender<<<1, 1>>>(d_renderer, d_camera, numSamples, numBounces,
                      d_rand_state);
  checkCudaError(cudaDeviceSynchronize());

  // CUDA

  int NUM_THREADS_X = 8;
  int NUM_THREADS_Y = 8;

  dim3 blocks(outputWidth / NUM_THREADS_X + 1,
              outputHeight / NUM_THREADS_Y + 1);
  dim3 threads(NUM_THREADS_X, NUM_THREADS_Y);

  p_render<<<blocks, threads>>>(d_scene, d_renderer, this->fb, d_rand_state);
}

__device__ Ray DRenderer::getRay(int i, int j,
                                 curandState *local_rand_state) const {
  auto pixel_center = pixel00 + i * pixelDeltaU + j * pixelDeltaV;
  auto pixel_sample = pixel_center + getPixelSampleSquare(local_rand_state);

  return Ray(center, pixel_sample - center);
}

__device__ Point3
DRenderer::getPixelSampleSquare(curandState *local_rand_state) const {
  auto x = -0.5 + random_float(local_rand_state);
  auto y = -0.5 + random_float(local_rand_state);

  return x * pixelDeltaU + y * pixelDeltaV;
}

__device__ Color DRenderer::getRayColor(const Ray &ray, Scene **scene,
                                        int num_bounces,
                                        curandState *local_rand_state) const {
  Ray r = ray;
  Color currentAttenuation(1.0, 1.0, 1.0);

  for (int i = 0; i < num_bounces; ++i) {
    HitRecord rec;

    if ((*scene)->hit(r, Interval(0, infinity), rec)) {
      Ray scattered;
      Color attenuation;

      if (rec.material->scatter(r, rec, attenuation, scattered,
                                local_rand_state)) {
        currentAttenuation *= attenuation;
        r = scattered;
      } else {
        return Vec3(0.0, 0.0, 0.0);
      }
    } else {
      Vec3 unit_direction = r.direction.normalize();
      auto a = 0.5 * (unit_direction.y + 1.0);
      Color color = (1.0 - a) * Color(1, 1, 1) + a * Color(0.5, 0.7, 1.0);

      return color * currentAttenuation;
    }
  }

  return Color();
}
