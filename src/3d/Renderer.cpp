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
int Renderer::getNumSamples() const { return _num_samples; }
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

void Renderer::setNumSamples(int num_samples) { _num_samples = num_samples; }

void Renderer::render(const Scene &scene) {
  for (int j = 0; j < _output_height; ++j) {
    for (int i = 0; i < _output_width; ++i) {
      Color pixel_color;

      for (int sample = 0; sample < _num_samples; ++sample) {
        Ray ray = getRay(i, j);
        pixel_color += getRayColor(ray, scene);
      }

      // Process samples

      pixel_color /= _num_samples;

      auto r = static_cast<float>(intensity.clamp(pixel_color.getX()));
      auto g = static_cast<float>(intensity.clamp(pixel_color.getY()));
      auto b = static_cast<float>(intensity.clamp(pixel_color.getZ()));

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

Color Renderer::getRayColor(const Ray &ray, const Scene &scene) const {
  if (HitRecord rec; scene.hit(ray, 0.001, INFINITY, rec)) {
    return 0.5 * (rec.normal + Color(1, 1, 1));
  }

  // Miss shader

  Vec3 unit_direction = unitVectorFrom(ray.getDirection());
  auto a = 0.5 * (unit_direction.getY() + 1.0);

  return (1.0 - a) * Color(1, 1, 1) + a * Color(0.5, 0.7, 1.0);
}
