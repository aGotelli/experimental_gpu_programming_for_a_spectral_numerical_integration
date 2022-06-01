#ifndef SPECTRAL_INTEGRATION_LIBRARY_H
#define SPECTRAL_INTEGRATION_LIBRARY_H

#include <Eigen/Dense>
#include <fstream>
#include <memory>
#include "odeBase.h"


constexpr unsigned int na = 3;  //  Kirkhoff rod
constexpr unsigned int ne = 3;  // dimesion of qe
constexpr unsigned int num_ch_nodes = 11;

template <unsigned int t_stateDim>
const Eigen::MatrixXd solveLinSys(std::shared_ptr<odeBase<t_stateDim, num_ch_nodes>> base) {
    
    constexpr int probDim = t_stateDim * num_ch_nodes;
    constexpr int unknownDim = t_stateDim * (num_ch_nodes-1);
    const double alpha = -1;
    const double beta = 0;
    const double beta_pos = 1;

    CUBLAS_CHECK(cublasDgemv(base->cublasH, CUBLAS_OP_N, 
                             unknownDim, t_stateDim, 
                             &alpha, 
                             base->d_D_IN, unknownDim, 
                             base->d_x0, 1, 
                             &beta_pos, 
                             base->d_b_NN, 1));
    CUBLAS_CHECK(cublasDgeam(base->cublasH, 
                             CUBLAS_OP_N, 
                             CUBLAS_OP_N, 
                             unknownDim, 
                             unknownDim, 
                             &alpha, 
                             base->d_A_NN, 
                             unknownDim, 
                             &beta_pos,
                             base->d_D_NN,
                             unknownDim,
                             base->d_A_NN,
                             unknownDim));

    std::vector<int> Ipiv(unknownDim, 0);
    int *d_Ipiv = nullptr; /* pivoting sequence */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int) * Ipiv.size()));
    const int pivot_on = 1;

    if (pivot_on) {
        CUSOLVER_CHECK(cusolverDnDgetrf(base->cusolverH, 
                                        unknownDim, 
                                        unknownDim, 
                                        base->d_A_NN, 
                                        unknownDim, 
                                        base->d_work, 
                                        d_Ipiv, 
                                        base->d_info));
        CUSOLVER_CHECK(cusolverDnDgetrs(base->cusolverH, 
                                        CUBLAS_OP_N, 
                                        unknownDim, 
                                        1, 
                                        base->d_A_NN, 
                                        unknownDim, 
                                        d_Ipiv, 
                                        base->d_b_NN, 
                                        unknownDim, 
                                        base->d_info));
    } else {
        CUSOLVER_CHECK(cusolverDnDgetrf(base->cusolverH, 
                                        unknownDim, 
                                        unknownDim, 
                                        base->d_A_NN, 
                                        unknownDim, 
                                        base->d_work, 
                                        NULL, 
                                        base->d_info));
        CUSOLVER_CHECK(cusolverDnDgetrs(base->cusolverH, 
                                        CUBLAS_OP_N, 
                                        unknownDim, 
                                        1, 
                                        base->d_A_NN, 
                                        unknownDim, 
                                        NULL, 
                                        base->d_b_NN, 
                                        unknownDim, 
                                        base->d_info));
    }

    Eigen::Matrix<double, unknownDim, 1> X_NN;
    Eigen::Matrix<double, t_stateDim, 1> initState;

    CUDA_CHECK( 
        cudaMemcpyAsync(X_NN.data(), base->d_b_NN, sizeof(double) * unknownDim, cudaMemcpyDeviceToHost, base->stream)
    );

    CUDA_CHECK(
        cudaMemcpyAsync(initState.data(), base->d_x0, sizeof(double) * t_stateDim, cudaMemcpyDeviceToHost, base->stream)
    );

    CUDA_CHECK(cudaStreamSynchronize(base->stream));

    Eigen::Matrix<double, probDim, 1> X_tilde;
    X_tilde << initState, X_NN;
    X_tilde = base->P*X_tilde;
    Eigen::Matrix<double, num_ch_nodes, t_stateDim> X_stack = Eigen::Map<Eigen::Matrix<double, num_ch_nodes, t_stateDim  >>(X_tilde.data());

    CUDA_CHECK(cudaFree(d_Ipiv));
    return X_stack;
}

template <unsigned int t_stateDim>
const Eigen::MatrixXd integrateODE(std::shared_ptr<odeBase<t_stateDim, num_ch_nodes>> base) {

    constexpr int probDim = t_stateDim * num_ch_nodes;
    constexpr int unknownDim = t_stateDim * (num_ch_nodes-1);

    const double alpha = 1;
    const double beta = 0;
    const double beta_pos = 1;
    
    CUBLAS_CHECK(cublasDgemv(base->cublasH, CUBLAS_OP_N, probDim, probDim, &alpha, base->d_P, probDim, base->d_b, 1, &beta, base->d_b, 1));
    CUBLAS_CHECK(cublasDgemm(
        base->cublasH,
        CUBLAS_OP_N, CUBLAS_OP_N,
        probDim, probDim, probDim,
        &alpha,
        base->d_A, probDim,
        base->d_P, probDim,
        &beta,
        base->d_A, probDim
    ));

    CUBLAS_CHECK(cublasDgemm(
        base->cublasH,
        CUBLAS_OP_T, CUBLAS_OP_N,
        probDim, probDim, probDim,
        &alpha,
        base->d_P, probDim,
        base->d_A, probDim,
        &beta,
        base->d_A, probDim
    ));

    CUBLAS_CHECK(cublasDcopy(
        base->cublasH, unknownDim,
        base->d_b+t_stateDim, 1,
        base->d_b_NN, 1
    ));

    for (unsigned int i = 0; i < t_stateDim; ++i) {
        CUBLAS_CHECK(cublasDcopy(
            base->cublasH, unknownDim,
            base->d_Dp+t_stateDim+i*probDim, 1,
            base->d_D_IN+i*unknownDim, 1
        ));

    }

    for (unsigned int i = 0; i < unknownDim; ++i) {
        CUBLAS_CHECK(cublasDcopy(
            base->cublasH, unknownDim,
            base->d_Dp+t_stateDim+t_stateDim*probDim+i*probDim, 1,
            base->d_D_NN+i*unknownDim, 1
        ));
    }

    for (unsigned int i = 0; i < unknownDim; ++i) {
        CUBLAS_CHECK(cublasDcopy(
            base->cublasH, unknownDim,
            base->d_A+t_stateDim+t_stateDim*probDim+i*probDim, 1,
            base->d_A_NN+i*unknownDim, 1
        ));
    }


    return solveLinSys<t_stateDim>(base);
}

#endif
