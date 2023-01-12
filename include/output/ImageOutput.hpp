#ifndef IMAGE_OUTPUT_H
#define IMAGE_OUTPUT_H

class ImageOutput {
public:
  virtual inline void writeColor(int r, int g, int b) const = 0;
};

#endif // IMAGE_OUTPUT_H