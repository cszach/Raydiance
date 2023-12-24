#include "Renderer.hpp"
#include "MathUtils.hpp"

Renderer::Renderer(Camera &camera, int output_width, int output_height)
    : _camera(camera) {
  setOutputSize(output_width, output_height); // Must set before setCamera
  setCamera(camera);
}

Camera &Renderer::getCamera() const { return _camera; }
int Renderer::getOutputWidth() const { return _output_width; }
int Renderer::getOutputHeight() const { return _output_height; }
const std::vector<float> &Renderer::getFrameBuffer() const {
  return _frame_buffer;
}

void Renderer::setCamera(const Camera &camera) {
  _camera = camera;

  auto viewport_u = camera.getViewportU();
  auto viewport_v = camera.getViewportV();

  _pixel_delta_u = viewport_u / _output_width;
  _pixel_delta_v = viewport_v / _output_height;

  _pixel00 =
      camera.getViewportUpperLeft() + 0.5 * (_pixel_delta_u + _pixel_delta_v);

  _center = camera.getPosition();
}

void Renderer::setOutputSize(int output_width, int output_height) {
  _output_width = output_width;
  _output_height = output_height;

  _num_pixels = output_width * output_height;
  _frame_buffer = std::vector<float>(3 * _num_pixels);
}

void Renderer::render(const Scene &scene) {
  for (int j = 0; j < _output_height; ++j) {
    for (int i = 0; i < _output_width; ++i) {
      Color pixel_color;

      for (int sample = 0; sample < num_samples; ++sample) {
        Ray ray = getRay(i, j);
        pixel_color += getRayColor(ray, num_bounces, scene);
      }

      // Process samples

      pixel_color /= num_samples;

      auto r = static_cast<float>(intensity.clamp(pixel_color.x));
      auto g = static_cast<float>(intensity.clamp(pixel_color.y));
      auto b = static_cast<float>(intensity.clamp(pixel_color.z));

      //  Write color

      int pixel_index = 3 * (i + j * _output_width);

      _frame_buffer[pixel_index] = r;
      _frame_buffer[pixel_index + 1] = g;
      _frame_buffer[pixel_index + 2] = b;
    }
  }
}

// PRIVATE

Ray Renderer::getRay(int i, int j) const {
  auto pixel_center = _pixel00 + i * _pixel_delta_u + j * _pixel_delta_v;
  auto pixel_sample = pixel_center + getPixelSampleSquare();

  return Ray(_center, pixel_sample - _center);
}

Point3 Renderer::getPixelSampleSquare() const {
  auto x = -0.5 + random_double();
  auto y = -0.5 + random_double();

  return x * _pixel_delta_u + y * _pixel_delta_v;
}

Color Renderer::getRayColor(const Ray &ray, int depth,
                            const Scene &scene) const {
  if (depth <= 0) {
    return Color();
  }

  if (HitRecord rec; scene.hit(ray, 0.001, INFINITY, rec)) {
    Vec3 direction = rec.normal + Vec3::randomUnit();
    return 0.5 * getRayColor(Ray(rec.p, direction), depth - 1, scene);
  }

  // Miss shader

  Vec3 unit_direction = ray.getDirection().normalize();
  auto a = 0.5 * (unit_direction.y + 1.0);

  return (1.0 - a) * Color(1, 1, 1) + a * Color(0.5, 0.7, 1.0);
}
