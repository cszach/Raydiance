# Raydiance

![CMake workflow status](https://github.com/cszach/Raydiance/actions/workflows/cmake.yml/badge.svg)

[blog]: https://cszach.github.io/Raydiance/

This is hopefully will be my long-term ray tracing software project. I will keep
adding features to the ray tracer as I learn and discover ray tracing
techniques.

## Goal

At the moment, I see two primary goals for this project:

1. Declare a public ray tracing API so external applications may include it; and
2. Create an interactive application window (perhaps using ImGui) to demonstrate
   the ray tracing API.

## Current status

Graphics hello world.

![Current output image.](image.ppm)

## Build

Requirements:

- CMake version 3.0+
- GCC (will add support for other compilers in the future)

```
cmake -B build .
cmake --build build
```

The binary will be in the `bin` folder.

## Test

After building the program, you can run unit tests by doing:

```bash
cd build
ctest
```

Or, if you want to see a more verbose output:

```bash
cd bin
./raydiance_test
```

## Blog

I am setting up a development blog for this project. When it is done, it should
live [here][blog].
