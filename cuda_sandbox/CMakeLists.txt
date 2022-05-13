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

# ##########################################
# cusolver_examples target
# ##########################################
include_directories(include)


add_executable(${PROJECT_NAME}_matrix_multiplication 
		src/matrix_multiplication/example_matrix_multiplication.cu
                include/tictoc.h)
target_include_directories(${PROJECT_NAME}_matrix_multiplication
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_matrix_multiplication
        PUBLIC
        cublas
        Eigen3::Eigen
)
#set_target_properties(${PROJECT_NAME}_matrix_multiplication PROPERTIES CUDA_ARCHITECTURES OFF)







add_executable(${PROJECT_NAME}_matrix_inverse 
		src/getrf/cusolver_getrf_example.cu
		include/tictoc.h)
target_include_directories(${PROJECT_NAME}_matrix_inverse
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_matrix_inverse
    PUBLIC
        cusolver
        cublas
        cublasLt
        cusparse
        cusolverMg
        Eigen3::Eigen
)
#set_target_properties(${PROJECT_NAME}_matrix_inverse PROPERTIES CUDA_ARCHITECTURES OFF)





#+++++++++++++++++++++++++++
#       DOXYGEN            +
#+++++++++++++++++++++++++++
# Think to document also GPU code
#if(BUILD_DOXYGEN_DOCUMENTATION)
#find_package(Doxygen)
#    if(NOT DOXYGEN_FOUND)
#        message(FALTAL_ERROR "Doxygen documentation generation requested ( BUILD_DOXYGEN_DOCUMENTATION=ON) but Doxygen is not installed.")
#    endif()
#
#    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
#    set(doxyfile    ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
#
#    configure_file(${doxyfile_in} ${doxyfile} @ONLY)
#
#    add_custom_target(doc
#        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
#        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#        COMMENT "Generating API documentation with Doxygen"
#        VERBATIM)
#
#   install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc/html DESTINATION hydra/doc/doxygen)
#
#endif(BUILD_DOXYGEN_DOCUMENTATION)




# ##########################################
# cusolver_examples directories
# ##########################################

# By default put binaries in build/bin (pre-install)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Installation directories
#set(CUSOLVER_EXAMPLES_BINARY_INSTALL_DIR "cusolver_examples/bin")

# ##########################################
# Install examples
# ##########################################

#IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
#  SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH "" FORCE)
#ENDIF()
