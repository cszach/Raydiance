#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <limits>

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

inline double degToRad(double deg) { return deg * pi / 180.0; }
inline double clamp(double value, double min, double max) {
  if (value < min)
    return min;
  if (value > max)
    return max;
  return value;
}

#endif // MATH_UTILS_H