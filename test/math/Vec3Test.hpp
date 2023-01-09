#include <gtest/gtest.h>

#include "Vec3.hpp"

class Vec3Test : public ::testing::Test {
protected:
  Vec3 u;
  Vec3 v;

  void SetUp() override {
    u = Vec3(0, 1, 2);
    v = Vec3(3, 4, 5);
  }
};

TEST_F(Vec3Test, GettersAndSetters) {
  EXPECT_EQ(u.getX(), 0);
  EXPECT_EQ(u.getY(), 1);
  EXPECT_EQ(u.getZ(), 2);

  u.setX(3);
  u.setY(4);
  u.setZ(5);

  EXPECT_EQ(u.getX(), 3);
  EXPECT_EQ(u.getY(), 4);
  EXPECT_EQ(u.getZ(), 5);
}

TEST_F(Vec3Test, BasicOperators) {
  EXPECT_EQ(u.lengthSquared(), 5);
  EXPECT_EQ(u.length(), sqrt(5));
  EXPECT_EQ(v.lengthSquared(), 50);
  EXPECT_EQ(v.length(), sqrt(50));

  EXPECT_TRUE(u == Vec3(0, 1, 2));
  EXPECT_TRUE(u != v);

  EXPECT_EQ(-u, Vec3(0, -1, -2));
  EXPECT_EQ(-v, Vec3(-3, -4, -5));

  EXPECT_EQ(u += v, Vec3(3, 5, 7));
  EXPECT_EQ(u -= v, Vec3(0, 1, 2));
  EXPECT_EQ(u *= v, Vec3(0, 4, 10));
  EXPECT_EQ(u /= 1, Vec3(0, 4, 10));
  EXPECT_EQ(u /= 2, Vec3(0, 2, 5));

  // mutability: the above operations mutate u

  EXPECT_EQ(u, Vec3(0, 2, 5));
  EXPECT_EQ(v, Vec3(3, 4, 5));
}

TEST_F(Vec3Test, Utilities) {
  EXPECT_EQ(u + v, Vec3(3, 5, 7));
  EXPECT_EQ(u - v, Vec3(-3, -3, -3));
  EXPECT_EQ(u * v, Vec3(0, 4, 10));
  EXPECT_EQ(u / 1.0, Vec3(0, 1, 2));
  EXPECT_EQ(u / 2.0, Vec3(0, 0.5, 1));

  EXPECT_EQ(dotProduct(u, v), 14);
  EXPECT_EQ(crossProduct(u, v), Vec3(-3, 6, -3));

  EXPECT_EQ(unitVectorFrom(u), Vec3(0, 1, 2) / sqrt(5));

  // immutability

  EXPECT_EQ(u, Vec3(0, 1, 2));
  EXPECT_EQ(v, Vec3(3, 4, 5));
}