#include <fstream>
#include <ostream>

#include "PPMOutput.hpp"
#include "Renderer.hpp"

int main() {
  // Image

  const int IMAGE_WIDTH = 256;
  const int IMAGE_HEIGHT = 256;

  std::ofstream f_out("image.ppm");
  std::ostream &f_ostream(f_out);

  f_ostream.rdbuf(f_out.rdbuf());

  auto pOutput =
      make_shared<PPMOutput>(IMAGE_WIDTH, IMAGE_HEIGHT, 255, f_ostream);

  pOutput->writeHeader();

  Renderer renderer(pOutput);

  renderer.render(Scene(), Camera());
}