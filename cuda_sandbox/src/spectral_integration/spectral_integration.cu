#include <iostream>
#include <cuda_runtime.h>
#include <cublas.h>
#include <utility>

#include <cusolverDn.h>

#include "cusolver_utils.h"

#include "tictoc.h" //  Mesuring runtime

#include <Eigen/Dense>

#include <fstream>
#include <vector>

#include <iostream>

/*! \file main.cpp
    \brief The main file performing the spectral numerical integration.

    In this file, we perform the computation from the PDF.
*/
#include <algorithm>
#include <numeric>
#include <cmath>

#include "chebyshev_differentiation.h"
#include "spectral_integration_utilities.h"
#include "lie_algebra_utilities.h"
#include "spectral_integration_library.h"


const auto chebyshev_points_top_down = ComputeChebyshevPoints<num_ch_nodes>(TOP_TO_BOTTOM);
const auto chebyshev_points_bottom_up = ComputeChebyshevPoints<num_ch_nodes>(BOTTOM_TO_TOP);
const std::array<std::array<double, num_ch_nodes>, 2> chebyshev_points = {chebyshev_points_bottom_up, chebyshev_points_top_down};

const auto Phi_top_down = Phi<na, ne, num_ch_nodes>(chebyshev_points[TOP_TO_BOTTOM]);
const auto Phi_bottom_up = Phi<na, ne, num_ch_nodes>(chebyshev_points[BOTTOM_TO_TOP]);
const std::array<std::array<Eigen::MatrixXd, num_ch_nodes>, 2> Phi_matrix = {Phi_bottom_up, Phi_top_down};

// //density of iron [kg/m^3]
// constexpr double rodDensity =  7874;

// //1cm radius
// const double rodCrossSec = M_PI * pow(0.01, 2);

// //gravitational acceleration
// constexpr double g = 9.8067;

template<unsigned int t_stateDimension>
Eigen::MatrixXd getQuaternionA(Eigen::VectorXd &t_qe) {
    constexpr integrationDirection direction = BOTTOM_TO_TOP;

    constexpr unsigned int probDimension = t_stateDimension*num_ch_nodes;

    Eigen::Vector3d K;
    Eigen::Matrix<double, t_stateDimension, t_stateDimension> A_at_chebyshev_point;
    //  Declare the matrix for the system Ax = b
    Eigen::Matrix<double, probDimension, probDimension> A =
            Eigen::Matrix<double, probDimension, probDimension>::Zero();

    for(unsigned int i=0; i < num_ch_nodes; i++){

        //  Extract the curvature from the strain
        K = Phi_matrix[direction][i]*t_qe;

        //  Compute the A matrix of Q' = 1/2 A(K) Q
        A_at_chebyshev_point <<      0, -K(0),  -K(1),  -K(2),
                                  K(0),     0,   K(2),  -K(1),
                                  K(1), -K(2),      0,   K(0),
                                  K(2),  K(1),  -K(0),      0;

        A_at_chebyshev_point = 0.5*A_at_chebyshev_point;

        for (unsigned int row = 0; row < A_at_chebyshev_point.rows(); ++row) {
            for (unsigned int col = 0; col < A_at_chebyshev_point.cols(); ++col) {
                int row_index = row*num_ch_nodes+i;
                int col_index = col*num_ch_nodes+i;
                A(row_index, col_index) = A_at_chebyshev_point(row, col);
            }
        }
    }

    return A;
}


int main(int argc, char *argv[]) {
    //Quaternion

    constexpr int qStateDim = 4;
    constexpr int qUnknownDim = qStateDim*(num_ch_nodes-1);
    constexpr int qProbDim = qStateDim*num_ch_nodes;

    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    cusolverDnHandle_t cusolverH = NULL;

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    Eigen::VectorXd qe(9);
    //  Here we give some value for the strain

    qe <<   0,
            0,
            0,
            1.28776905384098,
           -1.63807577049031,
            0.437404540900837,
            0,
            0,
            0;

    const Eigen::Vector4d initQuaternion(1, 0, 0, 0);
    Eigen::Matrix<double, qProbDim, qProbDim> q_A = getQuaternionA<qStateDim>(qe);
    Eigen::Matrix<double, qProbDim, 1> q_b = Eigen::Matrix<double, qProbDim, 1>::Zero();
    Eigen::Matrix<double, num_ch_nodes, num_ch_nodes> q_D_N = getDn<num_ch_nodes>(BOTTOM_TO_TOP);
    Eigen::Matrix<double, qProbDim, qProbDim> q_D = kroneckerProduct<qProbDim, qProbDim>(Eigen::Matrix<double, qStateDim, qStateDim>::Identity(), q_D_N);
    Eigen::Matrix<double, qProbDim, qProbDim> q_P = getP<qStateDim, num_ch_nodes>();

    Eigen::Matrix<double, qUnknownDim, 1> q_b_NN = Eigen::Matrix<double, qUnknownDim, 1>::Zero();
    Eigen::Matrix<double, qUnknownDim, qStateDim> q_D_IN;
    q_D_IN << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, -0.585786, 0, 0,
            0, 1, 0, 0,
            0, -3.41421, 0, 0,
            -0.585786, 0, 0, 0,
            0, 0, 1, 0,
            0, 0 ,-0.585786, 0,
            0, 0, 1, 0,
            0, 0  ,-3.41421, 0,
            1, 0, 0, 0,
            0, 0, 0, 1,
            0, 0, 0 ,-0.585786,
            0, 0, 0, 1,
            0, 0, 0  ,-3.41421,
            -3.41421, 0, 0, 0;

    Eigen::Matrix<double, qUnknownDim, qUnknownDim> q_A_NN;
    q_A_NN << 0, 0, 0, 0, 0, 0 ,-0.0435489, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,-0.0435489, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  ,-0.119413, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  ,-0.534533, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0   ,-1.27771, 0,
            0, 0, 0, 0, 0, 0, 0  ,-0.119413, 0, 0, 0, 0, 0, 0, 0, 0,
            0.0435489, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0  ,0.119413, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0  ,0.534533, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,1.27771,
            0, 0, 0, 0, 0, 0, 0, 0  ,-0.534533, 0, 0, 0, 0, 0, 0, 0,
            0, 0.0435489, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0,  0.119413, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,  0.534533, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,   1.27771, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,  -1.27771, 0, 0, 0, 0, 0, 0;

    Eigen::Matrix<double, qUnknownDim, qUnknownDim> q_D_NN;
    
    q_D_NN << 11,  0,  0,  0,  0, -13.6569,  0,  0,  0,  0,  4,  0,  0,  0,  0, -2.34315,
            0, 11, -13.6569,  4, -2.34315,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  3.41421, -1.41421, -2.82843,  1.41421,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,
            0, -1,  2.82843,  0, -2.82843,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0, 0.585786, -1.41421,  2.82843,  1.41421,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,
            3.41421,  0,  0,  0,  0, -1.41421,  0,  0,  0,  0, -2.82843,  0,  0,  0,  0, 1.41421,
            0,  0,  0,  0,  0,  0, 11, -13.6569,  4, -2.34315,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  3.41421, -1.41421, -2.82843,  1.41421,  0,  0,  0,  0, 0,  0,
            0,  0,  0,  0,  0,  0, -1,  2.82843,  0, -2.82843,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0, 0.585786, -1.41421,  2.82843,  1.41421,  0,  0,  0,  0, 0,  0,
            -1,  0,  0,  0,  0,  2.82843,  0,  0,  0,  0,  0,  0,  0,  0,  0, -2.82843,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 11, -13.6569,  4, -2.34315,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  3.41421, -1.41421, -2.82843, 1.41421,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  2.82843,  0, -2.82843,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0.585786, -1.41421,  2.82843, 1.41421,  0,
            0.585786,  0,  0,  0,  0, -1.41421,  0,  0,  0,  0,  2.82843,  0,  0,  0,  0, 1.41421;


    double* h_A = q_A.data();
    double* h_b = q_b.data();
    double* h_D = q_D.data();
    double* h_P = q_P.data();

    double* d_b_NN = nullptr;
    double* d_D_IN = nullptr;
    double* d_x0 = nullptr;
    double* d_A_NN = nullptr;
    double* d_D_NN = nullptr;
    double* d_A = nullptr;
    double* d_b = nullptr;
    double* d_D = nullptr;
    double* d_P = nullptr;

    int info = 0;
    int *d_info = nullptr; /* error info */

    int lwork = 0;            /* size of workspace */
    double *d_work = nullptr; /* device workspace for getrf */

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_NN), sizeof(double) * qUnknownDim));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D_IN), sizeof(double) * qUnknownDim*qStateDim));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x0), sizeof(double) * qStateDim));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A_NN), sizeof(double) * qUnknownDim*qUnknownDim));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D_NN), sizeof(double) * qUnknownDim*qUnknownDim));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * qProbDim*qProbDim));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), sizeof(double) * qProbDim));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D), sizeof(double) * qProbDim*qProbDim));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_P), sizeof(double) * qProbDim*qProbDim));

    //CUDA_CHECK(cudaMemcpyAsync(d_b_NN, q_b_NN.data(), sizeof(double) * qUnknownDim, cudaMemcpyHostToDevice, stream));
    //CUDA_CHECK(cudaMemcpyAsync(d_D_IN, q_D_IN.data(), sizeof(double) * qUnknownDim*qStateDim, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_x0, initQuaternion.data(), sizeof(double) * qStateDim, cudaMemcpyHostToDevice, stream));
    //CUDA_CHECK(cudaMemcpyAsync(d_A_NN, q_A_NN.data(), sizeof(double) * qUnknownDim*qUnknownDim, cudaMemcpyHostToDevice, stream));
    //CUDA_CHECK(cudaMemcpyAsync(d_D_NN, q_D_NN.data(), sizeof(double) * qUnknownDim*qUnknownDim, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_info, &info, sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_A, q_A.data(), sizeof(double) * qProbDim*qProbDim, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_b, q_b.data(), sizeof(double) * qProbDim, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_D, q_D.data(), sizeof(double) * qProbDim*qProbDim, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_P, q_P.data(), sizeof(double) * qProbDim*qProbDim, cudaMemcpyHostToDevice, stream));

    CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(cusolverH, qUnknownDim, qUnknownDim, d_A_NN, qUnknownDim, &lwork));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

    Eigen::Matrix<double, num_ch_nodes, qStateDim> X_stack;
    X_stack = integrateODE<qStateDim>(cusolverH,
                                      cublasH,
                                      stream,
                                      d_b_NN,
                                      d_D_IN,
                                      d_x0,
                                      d_A_NN,
                                      d_D_NN,
                                      d_work,
                                      d_info,
                                      d_A,
                                      d_b,
                                      d_D,
                                      d_P);

    std::cout << X_stack << std::endl;

    CUDA_CHECK(cudaFree(d_b_NN));
    CUDA_CHECK(cudaFree(d_D_IN));
    CUDA_CHECK(cudaFree(d_x0));
    CUDA_CHECK(cudaFree(d_A_NN));
    CUDA_CHECK(cudaFree(d_D_NN));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_D));
    CUDA_CHECK(cudaFree(d_P));

    return 0;
}
