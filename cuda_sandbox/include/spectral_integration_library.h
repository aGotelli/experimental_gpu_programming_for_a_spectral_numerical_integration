#ifndef SPECTRAL_INTEGRATION_LIBRARY_H
#define SPECTRAL_INTEGRATION_LIBRARY_H

#include <Eigen/Dense>
#include <fstream>
#include "tictoc.h"
#include <memory>
#include "odeBase.h"
#include "qIntegrator.h"
#include "globals.h"

__global__ void block_copy(double* src, 
                           const int src_rows,
                           double* dst, 
                           const int dst_rows,
                           const int row_index,
                           const int col_index) {
    const unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int tidx = threadIdx.x;
    const unsigned int bidx = blockIdx.x;
    //blocks = columns
    //threads = rows

    dst[tidx+bidx*dst_rows] = src[row_index+col_index*src_rows+tidx+bidx*src_rows];
    // cublasDcopy(cublasH, dst_rows,
    //             src+row_index+col_index*src_rows+tid*src_rows, 1,
    //             dst+tid*dst_rows, 1);
}

template <unsigned int t_stateDim>
const Eigen::MatrixXd solveLinSys(qIntegrator<t_stateDim, num_ch_nodes>* base, 
                                    cublasHandle_t &t_cublasH,
                                    cusolverDnHandle_t &t_cusolverH) {
    
    constexpr int probDim = t_stateDim * num_ch_nodes;
    constexpr int unknownDim = t_stateDim * (num_ch_nodes-1);
    const double alpha = -1;
    const double beta = 0;
    const double beta_pos = 1;

    CUBLAS_CHECK(cublasDgemv(t_cublasH, CUBLAS_OP_N, 
                             unknownDim, t_stateDim, 
                             &alpha, 
                             base->d_D_IN, unknownDim, 
                             base->d_x0, 1, 
                             &beta_pos, 
                             base->d_b_NN, 1));
    CUBLAS_CHECK(cublasDgeam(t_cublasH, 
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
    const int pivot_on = 0;

    if (pivot_on) {
        CUSOLVER_CHECK(cusolverDnDgetrf(t_cusolverH, 
                                        unknownDim, 
                                        unknownDim, 
                                        base->d_A_NN, 
                                        unknownDim, 
                                        base->d_work, 
                                        d_Ipiv, 
                                        base->d_info));
        CUSOLVER_CHECK(cusolverDnDgetrs(t_cusolverH, 
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
        CUSOLVER_CHECK(cusolverDnDgetrf(t_cusolverH, 
                                        unknownDim, 
                                        unknownDim, 
                                        base->d_A_NN, 
                                        unknownDim, 
                                        base->d_work, 
                                        NULL, 
                                        base->d_info));
        CUSOLVER_CHECK(cusolverDnDgetrs(t_cusolverH, 
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
        cudaMemcpy(X_NN.data(), base->d_b_NN, sizeof(double) * unknownDim, cudaMemcpyDeviceToHost)
    );

    CUDA_CHECK(
        cudaMemcpy(initState.data(), base->d_x0, sizeof(double) * t_stateDim, cudaMemcpyDeviceToHost)
    );


    Eigen::Matrix<double, probDim, 1> X_tilde;
    X_tilde << initState, X_NN;
    X_tilde = base->P*X_tilde;
    Eigen::Matrix<double, num_ch_nodes, t_stateDim> X_stack = Eigen::Map<Eigen::Matrix<double, num_ch_nodes, t_stateDim  >>(X_tilde.data());

    CUDA_CHECK(cudaFree(d_Ipiv));
    return X_stack;
}

template <unsigned int t_stateDim>
const Eigen::MatrixXd integrateODE(qIntegrator<t_stateDim, num_ch_nodes>* base, 
                                    cublasHandle_t &t_cublasH,
                                    cusolverDnHandle_t &t_cusolverH) {

    constexpr int probDim = t_stateDim * num_ch_nodes;
    constexpr int unknownDim = t_stateDim * (num_ch_nodes-1);

    const double alpha = 1;
    const double beta = 0;
    const double beta_pos = 1;
    
    CUBLAS_CHECK(cublasDgemv(t_cublasH, CUBLAS_OP_N, probDim, probDim, &alpha, base->d_P, probDim, base->d_b, 1, &beta, base->d_b, 1));
    CUBLAS_CHECK(cublasDgemm(
        t_cublasH,
        CUBLAS_OP_N, CUBLAS_OP_N,
        probDim, probDim, probDim,
        &alpha,
        base->d_A, probDim,
        base->d_P, probDim,
        &beta,
        base->d_A, probDim
    ));


    CUBLAS_CHECK(cublasDgemm(
        t_cublasH,
        CUBLAS_OP_T, CUBLAS_OP_N,
        probDim, probDim, probDim,
        &alpha,
        base->d_P, probDim,
        base->d_A, probDim,
        &beta,
        base->d_A, probDim
    ));

    CUBLAS_CHECK(cublasDcopy(
        t_cublasH, unknownDim,
        base->d_b+t_stateDim, 1,
        base->d_b_NN, 1
    ));



    block_copy<<<t_stateDim, unknownDim>>>(base->d_Dp, probDim, base->d_D_IN, unknownDim, t_stateDim, 0);

    block_copy<<<unknownDim, unknownDim>>>(base->d_Dp, probDim, base->d_D_NN, unknownDim, t_stateDim, t_stateDim);


    block_copy<<<unknownDim, unknownDim>>>(base->d_A, probDim, base->d_A_NN, unknownDim, t_stateDim, t_stateDim);

   

    return solveLinSys<t_stateDim>(base, t_cublasH, t_cusolverH);

}

#endif
