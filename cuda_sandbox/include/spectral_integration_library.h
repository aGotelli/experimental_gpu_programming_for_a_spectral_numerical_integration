#ifndef SPECTRAL_INTEGRATION_LIBRARY_H
#define SPECTRAL_INTEGRATION_LIBRARY_H

#include <Eigen/Dense>
#include <fstream>
#include "tictoc.h"
#include <memory>
#include "odeBase.h"
#include "qIntegrator.h"
#include "globals.h"

template <unsigned int t_stateDim>
const double* solveLinSys(qIntegrator<t_stateDim, num_ch_nodes>* base, 
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
                             &beta, 
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

    const int pivot_on = 0;

    if (pivot_on) {
        CUSOLVER_CHECK(cusolverDnDgetrf(t_cusolverH, 
                                        unknownDim, 
                                        unknownDim, 
                                        base->d_A_NN, 
                                        unknownDim, 
                                        base->d_work, 
                                        base->d_Ipiv, 
                                        base->d_info));
        CUSOLVER_CHECK(cusolverDnDgetrs(t_cusolverH, 
                                        CUBLAS_OP_N, 
                                        unknownDim, 
                                        1, 
                                        base->d_A_NN, 
                                        unknownDim, 
                                        base->d_Ipiv, 
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

    return base->d_b_NN;
}

template <unsigned int t_stateDim>
const double* integrateODE(qIntegrator<t_stateDim, num_ch_nodes>* base, 
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

    block_copy<<<unknownDim, unknownDim>>>(base->d_A, probDim, base->d_A_NN, unknownDim, t_stateDim, t_stateDim);
    // for (unsigned int i = 0; i < unknownDim; ++i) {
    //     CUBLAS_CHECK(cublasDcopy(
    //         t_cublasH, unknownDim,
    //         base->d_A+t_stateDim+t_stateDim*probDim+i*probDim, 1,
    //         base->d_A_NN+i*unknownDim, 1
    //     ));
    // }
   
    return solveLinSys<t_stateDim>(base, t_cublasH, t_cusolverH);

}

#endif
