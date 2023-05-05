# ---[ Check cmake version.
cmake_minimum_required(VERSION 3.18.0 FATAL_ERROR)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
#set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/lib64)



# ---[ Project specification.
project(cuda_sandbox LANGUAGES CXX CUDA)

include(GNUInstallDirs)

# ##########################################
# cusolver_examples build mode
# ##########################################

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release")
else()
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

# ##########################################
# cusolver_examples building flags
# ##########################################

# Global CXX/CUDA flags

# Global CXX flags/options
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Global CUDA CXX flags/options
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

#                                       needed to pass lambdas to kernels
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  --expt-extended-lambda; # --opt-level=4 
#)

SET(CUDA_PROPAGATE_HOST_FLAGS ON)
SET(CUDA_SEPARABLE_COMPILATION OFF)
SET(CUDA_VERBOSE_BUILD ON)

# Debug options
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS} -O0 -g -lineinfo")

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

find_package(benchmark REQUIRED)

# ##########################################
# cusolver_examples target
# ##########################################
include_directories(include)


add_executable(${PROJECT_NAME}_matrix_copy
		src/copy/cublas_copy_example.cu)
target_include_directories(${PROJECT_NAME}_matrix_copy
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_matrix_copy
    PUBLIC
        cublas
)


add_executable(${PROJECT_NAME}_matrix_multiplication
		src/matrix_multiplication/example_matrix_multiplication.cu)
target_include_directories(${PROJECT_NAME}_matrix_copy
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_matrix_multiplication
    PUBLIC
        cublas
        Eigen3::Eigen
)
