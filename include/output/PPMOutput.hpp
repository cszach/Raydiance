#ifndef PPM_OUTPUT_H
#define PPM_OUTPUT_H

#include <iostream>

#include "ImageOutput.hpp"

class PPMOutput : public ImageOutput {
private:
  int width;
  int height;
  int max_color;
  std::ostream &output_stream;

public:
  PPMOutput(int width, int height, int max_color = 255,
            std::ostream &output_stream = std::cout)
      : output_stream(output_stream) {}

  int getWidth() const { return this->width; }
  int getHeight() const { return this->height; }
  int getMaxColor() const { return this->max_color; }
  std::ostream &getOutputStream() const { return this->output_stream; }

  void writeHeader() {
    this->output_stream << "P3\n"
                        << this->width << ' ' << this->height << '\n'
                        << max_color << '\n';
  }

  inline void writeColor(int r, int g, int b) const override {
    this->output_stream << r << ' ' << g << ' ' << b << '\n';
  }
};

#endif // PPM_OUTPUT_H