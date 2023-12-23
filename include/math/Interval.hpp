#ifndef INTERVAL_H
#define INTERVAL_H

#include "MathUtils.hpp"

class Interval {
public:
  double _min;
  double _max;

  Interval() : _min(+infinity), _max(-infinity) {}

  Interval(double min, double max) : _min(min), _max(max) {}

  bool contains(double x) const { return _min <= x && x <= _max; }

  bool surrounds(double x) const { return _min < x && x < _max; }

  double clamp(double x) const {
    if (x < _min)
      return _min;

    if (x > _max)
      return _max;

    return x;
  }

  static const Interval empty;
  static const Interval universe;
};

const Interval empty(+infinity, -infinity);
const Interval universe(-infinity, +infinity);

#endif