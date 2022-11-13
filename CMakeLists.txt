cmake_minimum_required(VERSION 3.16)

project(spectral_integration LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

find_package(benchmark)

include_directories(include)

add_executable(spectral_integration main.cpp
                                    include/tictoc.h
                                    include/chebyshev_differentiation.h
                                    include/spectral_integration_utilities.h
                                    include/lie_algebra_utilities.h
                                    include/spectral_integration_library.h)
target_link_libraries(spectral_integration Eigen3::Eigen)


add_executable(benchmark benchmark/benchmark.cpp
                                    include/tictoc.h
                                    include/chebyshev_differentiation.h
                                    include/spectral_integration_utilities.h
                                    include/lie_algebra_utilities.h
                                    include/spectral_integration_library.h)
target_link_libraries(benchmark Eigen3::Eigen benchmark::benchmark)
