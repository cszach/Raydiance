#include "BVH.cuh"

__device__ BVHNode::BVHNode(Object **objects, size_t start, size_t end,
                            curandState *randState) {
  int axis = floor(curand_uniform(randState) * 2);
  auto comparator = (axis == 0)   ? boxXCompare
                    : (axis == 1) ? boxYCompare
                                  : boxZCompare;
  size_t objectSpan = end - start;

  if (objectSpan == 1) {
    left = right = objects[start];
  } else if (objectSpan == 2) {
    if (comparator(objects[start], objects[start + 1])) {
      left = objects[start];
      right = objects[start + 1];
    } else {
      left = objects[start + 1];
      right = objects[start];
    }
  } else {
    sortPrimitives(objects, start, end, comparator);

    auto mid = start + objectSpan / 2;
    left = new BVHNode(objects, start, mid, randState);
    right = new BVHNode(objects, mid, end, randState);
  }

  boundingBox = AABB(left->boundingBox, right->boundingBox);
}

__device__ bool BVHNode::hit(const Ray &ray, Interval ray_t,
                             HitRecord &rec) const {
  if (!boundingBox.hit(ray, ray_t)) {
    return false;
  }

  bool hitLeft = left->hit(ray, ray_t, rec);
  bool hitRight =
      right->hit(ray, Interval(ray_t.min, hitLeft ? rec.t : ray_t.max), rec);

  return hitLeft || hitRight;
}

__device__ void BVHNode::computeBoundingBox() { return; }

__device__ void BVHNode::sortPrimitives(Object **objects, int start, int end,
                                        bool(comparator)(const Object *a,
                                                         const Object *b)) {
  for (int i = start + 1; i < end; i++) {
    Object *o = objects[i];
    int j = i - 1;

    while (comparator(o, objects[j]) && j >= 0) {
      objects[j + 1] = objects[j];
      j--;
    }
    objects[j + 1] = o;
  }
}

__device__ bool BVHNode::boxCompare(const Object *a, const Object *b,
                                    int axis) {
  return a->boundingBox.axis(axis).min < b->boundingBox.axis(axis).min;
}

__device__ bool BVHNode::boxXCompare(const Object *a, const Object *b) {
  return boxCompare(a, b, 0);
}

__device__ bool BVHNode::boxYCompare(const Object *a, const Object *b) {
  return boxCompare(a, b, 1);
}

__device__ bool BVHNode::boxZCompare(const Object *a, const Object *b) {
  return boxCompare(a, b, 2);
}
