cmake_minimum_required(VERSION 3.16)

project(spectral_integration LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

add_executable(spectral_integration main.cpp)
target_link_libraries(spectral_integration Eigen3::Eigen)

