Raydiance
========

[blog]: https://cszach.github.io/Raydiance/

This is hopefully will be my long-term ray tracing software project. I will keep
adding features to the ray tracer as I learn and discover ray tracing
techniques.

Goal
----

At the moment, I see two primary goals for this project:

1. Declare a public ray tracing API so external applications may include it; and
2. Create an interactive application window (perhaps using ImGui) to demonstrate
   the ray tracing API.

Current status
--------------

Hello world project with CMake set up.

Build
-----

Requirements:
- CMake version 3.0+
- GCC (will add support for other compilers in the future)

```
mkdir build
cd build
cmake ..
cmake --build ..
```

The binary will be in the `bin` folder.

> Note: I have no prior professional experience with C++ and C++ tools such as
> CMake. I am learning C++ as I build this project. If anything does not make
> sense to you, please file an issue. Thank you for your understanding and help.

Blog
----

I am setting up a development blog for this project. When it is done, it should
live [here][blog].