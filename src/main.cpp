#include <fstream>
#include <memory>
#include <ostream>

#include "PPMOutput.hpp"
#include "Renderer.hpp"
#include "Scene.hpp"
#include "Sphere.hpp"

using std::make_shared;
using std::shared_ptr;

int main() {
  // Image

  const int IMAGE_WIDTH = 256;
  const int IMAGE_HEIGHT = 256;
  const int MAX_COLOR = 255;

  std::ofstream f_out("image.ppm");
  std::ostream &f_ostream(f_out);

  f_ostream.rdbuf(f_out.rdbuf());

  auto output = PPMOutput(IMAGE_WIDTH, IMAGE_HEIGHT, MAX_COLOR, f_ostream);

  output.writeHeader();

  // Camera

  Camera camera(100, 1);

  // Scene

  Scene scene;
  shared_ptr<Sphere> test_sphere = make_shared<Sphere>(0.5);
  test_sphere->setPosition(Point3(0, 0, -1));

  scene.add(test_sphere);

  Renderer renderer(output);

  renderer.render(scene, camera);
}