cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0135 NEW)

project(Raydiance)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Google Test setup

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Build

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/math
    ${CMAKE_SOURCE_DIR}/include/3d
    )
add_executable(raydiance src/main.cpp)

# Google Test

enable_testing()

add_executable(
    raydiance_test
    test/main.cpp
)
target_link_libraries(
    raydiance_test
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(raydiance_test)